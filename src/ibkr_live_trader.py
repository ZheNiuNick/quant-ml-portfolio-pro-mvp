#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IBKR Live Trader (基于 ib_insync) - 同步版本，稳定可靠

功能：
- 自动获取实时价格（支持延迟行情）
- 查看当前持仓
- 计算交易量（target_shares - current_shares）
- 支持限价单（默认）和市价单（可选）
- 等待订单成交（同步 waitUntilFilled）
- 清晰的日志输出
- 对接 weights.parquet 结构

使用方法：
1. 安装依赖：pip install ib_insync
2. 在 IB Gateway / TWS 中开启 API（Paper Account）
3. 生成最新权重：python src/optimizer.py --optimize
4. 运行脚本（推荐：自动读取账户资金，只用90%）：
   python src/ibkr_live_trader.py \
       --weights outputs/portfolios/weights.parquet \
       --capital 0 \
       --capital-usage-ratio 0.90 \
       --ib-host 127.0.0.1 \
       --ib-port 7497 \
       --client-id 777 \
       --order-type LIMIT \
       --price-offset 0.001 \
       --market-data-type delayed

安全特性：
- 自动读取账户可用资金（BuyingPower），无需手动指定
- 默认只使用 90% 的资金（--capital-usage-ratio 0.90），留 10% 缓冲
- 如果指定 --capital，会自动检查是否超过可用资金
- 计算交易量时会检查是否超过可用资金，自动缩减订单
- 市场数据支持 real/delayed/delayed_frozen，且自动 fallback 到 midpoint / last close
- 默认使用限价单（LIMIT），价格偏移 0.1%（避免滑点）
- 支持 long-only 策略（weights 都是正数）
- 自动计算需要买入/卖出的股数（基于目标持仓 - 当前持仓）
- 订单状态会等待直到 Filled 或超时
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

try:
    from ib_insync import IB, Stock, LimitOrder, MarketOrder, ExecutionFilter
except ImportError:
    raise ImportError("Please install ib_insync: pip install ib_insync")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class IBKRLiveTrader:
    """基于 ib_insync 的 IBKR 实盘交易器（同步版本）"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 777,
        order_type: str = "LIMIT",
        price_offset: float = 0.001,
        market_data_type: str = "delayed",
    ):
        """
        Args:
            host: IB Gateway/TWS 主机地址
            port: 端口（7497=Paper, 7496=Live）
            client_id: 客户端 ID
            order_type: 订单类型 ("LIMIT" 或 "MKT")
            price_offset: 限价单价格偏移（相对于当前价格的百分比，0.001=0.1%）
            market_data_type: 行情类型 ("real", "delayed", "delayed_frozen")
        """
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id
        self.order_type = order_type.upper()
        self.price_offset = price_offset
        self.market_data_type = market_data_type.lower()
        self.market_data_type_code = {"real": 1, "delayed": 3, "delayed_frozen": 4}.get(self.market_data_type, 3)
        self.orders: Dict[str, any] = {}  # ticker -> Trade
        self.positions: Dict[str, float] = {}  # ticker -> current_shares
        
        # 交易收集器：自动收集所有执行和佣金信息
        self.executions: list = []  # 存储所有 Execution 对象
        self.commissions: Dict[str, any] = {}  # execId -> CommissionReport

    def connect(self):
        """连接到 IB Gateway/TWS（同步）"""
        logger.info(f"Connecting to IBKR {self.host}:{self.port} (clientId={self.client_id})...")
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        logger.info("[OK] Connected to IBKR")
        self.ib.reqMarketDataType(self.market_data_type_code)
        logger.info(f"[IBKR] Market data type set to {self.market_data_type} (code {self.market_data_type_code})")
        
        # 注册事件监听器，自动收集所有交易执行和佣金信息
        self._register_event_listeners()
        
        # 方法1：使用 IB.fills() 获取填充记录（推荐，包含完整的合约和执行信息）
        logger.info("Fetching fill records using IB.fills()...")
        try:
            ib_fills = self.ib.fills()
            logger.info(f"IB.fills() returned {len(ib_fills)} fills")
            
            # 用于去重的 execId 集合
            collected_exec_ids = set()
            
            for fill in ib_fills:
                try:
                    if hasattr(fill, 'contract') and fill.contract and fill.contract.secType == "STK":
                        if hasattr(fill, 'execution') and fill.execution:
                            exec_id = fill.execution.execId if hasattr(fill.execution, 'execId') else None
                            
                            # 避免重复
                            if exec_id and exec_id in collected_exec_ids:
                                continue
                            
                            exec_data = {
                                "reqId": 0,
                                "contract": fill.contract,
                                "execution": fill.execution,
                                "timestamp": time.time()
                            }
                            self.executions.append(exec_data)
                            if exec_id:
                                collected_exec_ids.add(exec_id)
                            
                            symbol = fill.contract.symbol if hasattr(fill.contract, 'symbol') else "UNKNOWN"
                            side = fill.execution.side if hasattr(fill.execution, 'side') else "UNKNOWN"
                            shares = fill.execution.shares if hasattr(fill.execution, 'shares') else 0
                            price = fill.execution.price if hasattr(fill.execution, 'price') else 0
                            logger.info(f"Collected fill: {symbol}, {side}, {shares} shares @ ${price}")
                            
                            # 同时收集佣金信息
                            if hasattr(fill, 'commissionReport') and fill.commissionReport:
                                report = fill.commissionReport
                                if exec_id:
                                    self.commissions[exec_id] = report
                                    commission = report.commission if hasattr(report, 'commission') else 0
                                    logger.info(f"  Commission: ${commission}")
                except Exception as e:
                    logger.warning(f"Error processing fill: {e}")
                    continue
        except Exception as e:
            logger.warning(f"IB.fills() failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 方法2：使用 IB.executions() 获取执行记录（补充，但可能缺少合约信息）
        logger.info("Fetching execution records using IB.executions()...")
        try:
            ib_executions = self.ib.executions()
            logger.info(f"IB.executions() returned {len(ib_executions)} executions")
            
            for exec_item in ib_executions:
                try:
                    exec_id = exec_item.execId if hasattr(exec_item, 'execId') else None
                    
                    # 如果已经通过 fills() 收集过，跳过
                    if exec_id and exec_id in collected_exec_ids:
                        continue
                    
                    # 尝试从 trades() 中查找对应的合约
                    contract = None
                    order_id = exec_item.orderId if hasattr(exec_item, 'orderId') else None
                    if order_id:
                        for trade in self.ib.trades():
                            if hasattr(trade, 'order') and hasattr(trade.order, 'orderId') and trade.order.orderId == order_id:
                                contract = trade.contract
                                break
                    
                    # 如果找到合约，存储执行记录
                    if contract and contract.secType == "STK":
                        exec_data = {
                            "reqId": 0,
                            "contract": contract,
                            "execution": exec_item,
                            "timestamp": time.time()
                        }
                        self.executions.append(exec_data)
                        if exec_id:
                            collected_exec_ids.add(exec_id)
                        
                        symbol = contract.symbol if hasattr(contract, 'symbol') else "UNKNOWN"
                        side = exec_item.side if hasattr(exec_item, 'side') else "UNKNOWN"
                        shares = exec_item.shares if hasattr(exec_item, 'shares') else 0
                        price = exec_item.price if hasattr(exec_item, 'price') else 0
                        logger.info(f"Collected execution: {symbol}, {side}, {shares} shares @ ${price}")
                except Exception as e:
                    logger.warning(f"Error processing execution: {e}")
                    continue
        except Exception as e:
            logger.warning(f"IB.executions() failed: {e}")
        
        # 方法3：请求历史执行记录（通过 reqExecutions 回调，获取所有已执行的交易）
        logger.info("Requesting all execution records via reqExecutions()...")
        initial_exec_count = len(self.executions)
        
        # 获取账户ID（用于 ExecutionFilter）
        account_id = None
        try:
            account_summary = self.ib.accountSummary()
            if account_summary:
                account_id = account_summary[0].account
                logger.info(f"Using account: {account_id}")
        except Exception as e:
            logger.warning(f"Could not get account ID: {e}")
        
        # 使用 ExecutionFilter 获取所有历史执行记录
        # 根据 ib_insync 文档，reqExecutions 需要 ExecutionFilter 来获取历史记录
        # 不设置任何过滤条件（或只设置账户），应该能获取所有历史记录
        try:
            exec_filter = ExecutionFilter()
            # 只设置 clientId 和账户，不设置时间/股票过滤，获取所有历史记录
            exec_filter.clientId = self.client_id
            if account_id:
                exec_filter.acctCode = account_id
            # 不设置 time, symbol, secType, side, exchange，表示获取所有记录
            logger.info(f"Requesting executions with filter: account={account_id}, clientId={self.client_id} (no time/symbol filter = all history)")
            self.ib.reqExecutions(0, exec_filter)
        except TypeError:
            # 如果 reqExecutions 不接受 ExecutionFilter 参数，尝试不使用过滤器
            logger.info("reqExecutions() does not accept ExecutionFilter, trying without filter...")
            try:
                self.ib.reqExecutions(0)
            except Exception as e:
                logger.warning(f"reqExecutions(0) failed: {e}")
        except Exception as e:
            logger.warning(f"reqExecutions with filter failed: {e}, trying without filter...")
            import traceback
            traceback.print_exc()
            # 如果使用过滤器失败，回退到不使用过滤器
            try:
                self.ib.reqExecutions(0)
            except Exception as e2:
                logger.warning(f"reqExecutions(0) also failed: {e2}")
        
        # 等待回调完成（增加等待时间，确保所有历史记录都被收集）
        # 注意：reqExecutions 可能会分批返回数据，需要等待足够长的时间
        max_wait_time = 15  # 最多等待 15 秒
        stable_count = 0  # 连续稳定次数
        for i in range(max_wait_time):
            self.ib.sleep(1)
            current_count = len(self.executions)
            if current_count > initial_exec_count:
                new_count = current_count - initial_exec_count
                logger.info(f"[{i+1}s] Collected {new_count} new executions from reqExecutions(), total: {current_count}")
                stable_count = 0  # 重置稳定计数
                initial_exec_count = current_count  # 更新初始计数
            else:
                stable_count += 1
                # 如果连续 3 秒没有新数据，可以提前退出
                if stable_count >= 3:
                    logger.info(f"No new executions for {stable_count} seconds, stopping wait")
                    break
        
        # 从 fills() 中加载佣金信息（如果还没有加载）
        if len(self.commissions) == 0:
            self._load_commissions_from_fills()
        
        logger.info(f"[OK] Total executions collected: {len(self.executions)}, commissions: {len(self.commissions)}")

    def disconnect(self):
        """断开连接（同步）"""
        if self.ib.isConnected():
            # 取消事件监听器
            self._unregister_event_listeners()
            self.ib.disconnect()
            logger.info("[OK] Disconnected from IBKR")
    
    def _register_event_listeners(self):
        """注册事件监听器，自动收集所有交易执行和佣金信息"""
        # 注册执行详情事件
        self.ib.execDetailsEvent += self._on_exec_details
        # 注册佣金报告事件
        self.ib.commissionReportEvent += self._on_commission
        logger.info("[OK] Event listeners registered (execDetailsEvent, commissionReportEvent)")
    
    def _unregister_event_listeners(self):
        """取消事件监听器"""
        try:
            self.ib.execDetailsEvent -= self._on_exec_details
            self.ib.commissionReportEvent -= self._on_commission
        except Exception:
            pass
    
    def _on_exec_details(self, reqId, contract, execution):
        """执行详情事件回调：收集所有执行记录"""
        try:
            if contract and contract.secType == "STK":
                # 存储执行记录（包含合约和执行信息）
                exec_data = {
                    "reqId": reqId,
                    "contract": contract,
                    "execution": execution,
                    "timestamp": time.time()
                }
                self.executions.append(exec_data)
                symbol = contract.symbol if hasattr(contract, 'symbol') else "UNKNOWN"
                side = execution.side if hasattr(execution, 'side') else "UNKNOWN"
                shares = execution.shares if hasattr(execution, 'shares') else 0
                price = execution.price if hasattr(execution, 'price') else 0
                logger.info(f"Collected execution: {symbol}, {side}, {shares} shares @ ${price}")
        except Exception as e:
            logger.warning(f"Error in execDetails callback: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_commission(self, trade, fill, report):
        """佣金报告事件回调：收集所有佣金信息"""
        try:
            if report and hasattr(report, 'execId') and report.execId:
                self.commissions[report.execId] = report
                exec_id = report.execId
                commission = report.commission if hasattr(report, 'commission') else 0
                logger.info(f"Collected commission: execId={exec_id}, commission=${commission}")
        except Exception as e:
            logger.warning(f"Error in commissionReport callback: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_commissions_from_fills(self):
        """从 fills() 中加载佣金信息（如果还没有加载）"""
        try:
            fills = self.ib.fills()
            for fill in fills:
                if hasattr(fill, 'commissionReport') and fill.commissionReport:
                    report = fill.commissionReport
                    if hasattr(report, 'execId') and report.execId:
                        exec_id = report.execId
                        # 如果还没有这个佣金记录，才添加
                        if exec_id not in self.commissions:
                            self.commissions[exec_id] = report
                            commission = report.commission if hasattr(report, 'commission') else 0
                            logger.info(f"Collected commission from fill: execId={exec_id}, commission=${commission}")
        except Exception as e:
            logger.warning(f"Error loading commissions from fills(): {e}")
    
    def get_trades(self) -> list:
        """
        获取所有标准化的交易记录
        
        Returns:
            list: 标准化的交易记录列表，每个记录包含：
                - time: 交易时间 (YYYY-MM-DD HH:MM:SS)
                - symbol: 股票代码
                - side: 交易方向 (BOT/BUY 或 SLD/SELL)
                - quantity: 交易数量
                - price: 交易价格
                - amount: 交易金额 (quantity * price)
                - commission: 佣金
                - status: 状态 (FILLED)
        """
        trades = []
        
        for exec_data in self.executions:
            try:
                contract = exec_data["contract"]
                execution = exec_data["execution"]
                
                # 跳过非股票
                if not contract or contract.secType != "STK":
                    continue
                
                # 提取执行信息
                exec_time = ""
                if hasattr(execution, 'time') and execution.time:
                    exec_time = execution.time.strftime("%Y-%m-%d %H:%M:%S")
                
                shares = float(execution.shares) if hasattr(execution, 'shares') and execution.shares else 0.0
                price = float(execution.price) if hasattr(execution, 'price') and execution.price else 0.0
                side = execution.side if hasattr(execution, 'side') else "UNKNOWN"
                symbol = contract.symbol if hasattr(contract, 'symbol') else "UNKNOWN"
                exec_id = execution.execId if hasattr(execution, 'execId') else ""
                
                # 获取佣金
                commission = 0.0
                if exec_id and exec_id in self.commissions:
                    comm_report = self.commissions[exec_id]
                    if hasattr(comm_report, 'commission') and comm_report.commission:
                        commission = float(comm_report.commission)
                
                # 标准化 side (BOT -> BUY, SLD -> SELL)
                side_normalized = "BUY" if side in ["BOT", "BUY"] else "SELL" if side in ["SLD", "SELL"] else side
                
                if shares > 0 and price > 0:
                    trades.append({
                        "time": exec_time,
                        "symbol": symbol,
                        "side": side_normalized,
                        "quantity": shares,
                        "price": price,
                        "amount": shares * price,
                        "commission": commission,
                        "status": "FILLED"
                    })
            except Exception as e:
                logger.warning(f"Skipping execution due to error: {e}")
                continue
        
        # 按时间排序（最新的在前）
        trades.sort(key=lambda x: x["time"] if x["time"] else "", reverse=True)
        
        logger.info(f"Returning {len(trades)} trades from {len(self.executions)} collected executions")
        if len(trades) == 0 and len(self.executions) > 0:
            logger.warning(f"Warning: {len(self.executions)} executions collected but 0 trades returned. Check execution data format.")
            # 添加调试信息
            if len(self.executions) > 0:
                sample = self.executions[0]
                logger.debug(f"Sample execution: contract={sample.get('contract')}, execution={sample.get('execution')}")
        
        return trades

    def get_account_buying_power(self) -> Optional[float]:
        """获取账户可用资金（Buying Power）- 同步版本"""
        try:
            if not self.ib.isConnected():
                logger.warning("Not connected to IBKR")
                return None

            # 使用同步 API
            account_summary = self.ib.accountSummary()

            if not account_summary:
                logger.warning("No account summary available")
                return None

            account_id = account_summary[0].account if account_summary else None

            def _parse(records, tag):
                for record in records:
                    if record.tag == tag and record.account == account_id:
                        try:
                            return float(record.value)
                        except ValueError:
                            continue
                return None

            buying_power = _parse(account_summary, "BuyingPower")
            if buying_power is not None:
                logger.info(f"Account Buying Power: ${buying_power:,.2f}")
                return buying_power

            net_liq = _parse(account_summary, "NetLiquidation")
            if net_liq is not None:
                logger.info(f"Account Net Liquidation: ${net_liq:,.2f} (using as buying power)")
                return net_liq

            logger.warning("Could not find BuyingPower or NetLiquidation in account summary")

            # 备用：尝试 accountValues
            try:
                account_values = self.ib.accountValues()
                if account_values:
                    buying_power = next(
                        (float(av.value) for av in account_values if av.tag == "BuyingPower"),
                        None,
                    )
                    if buying_power:
                        logger.info(f"Account Buying Power (fallback): ${buying_power:,.2f}")
                        return buying_power
            except Exception:
                pass

            return None
        except Exception as e:
            logger.error(f"Failed to get account buying power: {e}")
            return None

    def get_current_positions(self) -> Dict[str, float]:
        """获取当前持仓（ticker -> shares）- 同步版本"""
        logger.info("Fetching current positions...")
        positions = {}
        for pos in self.ib.positions():
            if pos.contract.secType == "STK":
                ticker = pos.contract.symbol
                shares = pos.position
                if abs(shares) > 1e-6:  # 忽略接近 0 的持仓
                    positions[ticker] = shares
                    logger.info(f"  {ticker}: {shares:.2f} shares")
        self.positions = positions
        logger.info(f"[OK] Found {len(positions)} positions")
        return positions

    @staticmethod
    def _calculate_midprice(ticker_data) -> Optional[float]:
        """根据 bid/ask 计算 mid price"""
        bid = getattr(ticker_data, "bid", None)
        ask = getattr(ticker_data, "ask", None)
        if bid and ask and bid > 0 and ask > 0:
            return (bid + ask) / 2
        delayed_bid = getattr(ticker_data, "delayedBid", None)
        delayed_ask = getattr(ticker_data, "delayedAsk", None)
        if delayed_bid and delayed_ask and delayed_bid > 0 and delayed_ask > 0:
            return (delayed_bid + delayed_ask) / 2
        return None

    def get_realtime_price(self, ticker: str) -> Optional[float]:
        """获取实时/延迟价格（last / midpoint / close）- 同步版本"""
        ticker_data = None
        contract = None
        try:
            if not self.ib.isConnected():
                logger.warning(f"  {ticker}: Not connected to IBKR")
                return None

            contract = Stock(ticker, "SMART", "USD")
            # 同步 API
            self.ib.qualifyContracts(contract)

            # 请求市场数据，允许使用延迟行情
            ticker_data = self.ib.reqMktData(contract, "", False, False)
            # 等待价格更新（同步 sleep）
            self.ib.sleep(1)

            price_sources = [
                ("last", getattr(ticker_data, "last", None)),
                ("mid", self._calculate_midprice(ticker_data)),
                ("close", getattr(ticker_data, "close", None)),
                ("delayedLast", getattr(ticker_data, "delayedLast", None)),
                ("delayedClose", getattr(ticker_data, "delayedClose", None)),
            ]

            for source, value in price_sources:
                if value and value > 0:
                    logger.info(f"  {ticker}: ${value:.2f} ({source})")
                    return value

            logger.warning(f"  {ticker}: No valid price data")
            return None
        except ConnectionError as e:
            logger.warning(f"  {ticker}: Connection lost - {e}")
            return None
        except Exception as e:
            logger.error(f"  {ticker}: Failed to get price - {e}")
            return None
        finally:
            if contract and self.ib.isConnected():
                try:
                    self.ib.cancelMktData(contract)
                except Exception:
                    pass

    def get_realtime_prices(self, tickers: list) -> Dict[str, float]:
        """批量获取实时价格 - 同步版本"""
        if not self.ib.isConnected():
            logger.error("[Error] Not connected to IBKR. Cannot fetch prices.")
            return {}

        logger.info(f"Fetching realtime prices for {len(tickers)} tickers...")
        prices = {}
        for ticker in tickers:
            if not self.ib.isConnected():
                logger.warning(f"[Warning] Connection lost during price fetching. Stopped at {ticker}.")
                break
            price = self.get_realtime_price(ticker)
            if price:
                prices[ticker] = price
        logger.info(f"[OK] Got prices for {len(prices)}/{len(tickers)} tickers")
        return prices

    def calculate_trade_shares(
        self,
        target_weights: pd.Series,
        prices: Dict[str, float],
        total_capital: float,
        available_cash: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        计算需要交易的股数（target_shares - current_shares）
        
        重要：确保卖出所有不在目标权重中的持仓，保证最终持仓数 = len(target_weights)

        Args:
            target_weights: 目标权重 Series (ticker -> weight)
            prices: 实时价格 Dict (ticker -> price)
            total_capital: 总资金
            available_cash: 可用现金（用于安全检查）

        Returns:
            Dict (ticker -> trade_shares)，正数=买入，负数=卖出
        """
        trades = {}
        total_buy_value = 0.0  # 累计买入金额（用于安全检查）
        target_tickers = set(target_weights[target_weights > 0].index)  # 目标持仓股票集合

        # 第一步：处理目标权重中的股票（买入/调整）
        for ticker, weight in target_weights.items():
            if weight <= 0:
                continue  # 跳过权重为 0 的股票

            price = prices.get(ticker)
            if not price or price <= 0:
                logger.warning(f"[Skip] {ticker}: No valid price")
                continue

            # 目标持仓价值
            target_value = total_capital * weight
            target_shares = target_value / price

            # 当前持仓
            current_shares = self.positions.get(ticker, 0.0)

            # 需要交易的股数
            trade_shares = target_shares - current_shares

            # 详细日志：显示换手情况
            if trade_shares < 0:
                logger.info(f"[Turnover] {ticker}: Need to SELL {abs(trade_shares):.2f} shares (current={current_shares:.2f}, target={target_shares:.2f})")
            elif trade_shares > 0:
                logger.debug(f"[Turnover] {ticker}: Need to BUY {trade_shares:.2f} shares (current={current_shares:.2f}, target={target_shares:.2f})")
            else:
                logger.debug(f"[Turnover] {ticker}: No trade needed (current={current_shares:.2f}, target={target_shares:.2f})")

            # 过滤掉交易量过小的订单（< 1 股或 < $100）
            # 注意：对于卖出订单，如果金额 < $100，仍然应该卖出（避免持仓过多）
            if abs(trade_shares) < 1.0:
                if trade_shares < 0:
                    logger.warning(f"[Skip] {ticker}: Sell order too small ({abs(trade_shares):.2f} shares < 1.0)")
                else:
                    logger.debug(f"[Skip] {ticker}: Trade too small ({trade_shares:.2f} shares < 1.0)")
                continue
            
            # 对于买入订单，如果金额 < $100，跳过（避免小额交易）
            # 对于卖出订单，即使金额 < $100，也允许卖出（清理持仓）
            if trade_shares > 0 and abs(trade_shares * price) < 100:
                logger.debug(f"[Skip] {ticker}: Buy order too small (${trade_shares * price:.2f} < $100)")
                continue

            # 安全检查：如果是买入，检查是否超过可用资金
            if trade_shares > 0 and available_cash is not None:
                buy_value = trade_shares * price
                if total_buy_value + buy_value > available_cash:
                    # 按比例缩减买入量
                    remaining_cash = available_cash - total_buy_value
                    if remaining_cash > 100:  # 至少保留 $100 缓冲
                        max_shares = remaining_cash / price
                        trade_shares = min(trade_shares, max_shares)
                        logger.warning(
                            f"[Safety] {ticker}: Reduced buy order to {trade_shares:.2f} shares "
                            f"(limited by available cash: ${remaining_cash:,.2f})"
                        )
                    else:
                        logger.warning(f"[Skip] {ticker}: Insufficient cash (need ${buy_value:,.2f}, have ${remaining_cash:,.2f})")
                        continue

            if trade_shares > 0:
                total_buy_value += trade_shares * price
            elif trade_shares < 0:
                # 卖出订单，不需要累计买入金额
                logger.info(f"[Turnover] {ticker}: Generating SELL order for {abs(trade_shares):.2f} shares")

            trades[ticker] = trade_shares
            action = "BUY" if trade_shares > 0 else "SELL"
            logger.info(
                f"[Trade] {ticker}: {action} {abs(trade_shares):.2f} shares "
                f"(current={current_shares:.2f}, target={target_shares:.2f}, price=${price:.2f})"
            )

        # 第二步：卖出所有不在目标权重中的持仓（关键修复）
        logger.info(f"\n[Step 2] Checking stocks to sell (not in target portfolio)...")
        stocks_not_in_target = []
        for ticker, current_shares in self.positions.items():
            if ticker not in target_tickers and abs(current_shares) > 1e-6:
                stocks_not_in_target.append(ticker)
                # 这个股票不在目标权重中，需要全部卖出
                price = prices.get(ticker)
                if not price or price <= 0:
                    logger.warning(f"[Skip] {ticker}: No valid price for selling (will try to get price)")
                    # 尝试获取价格
                    price = self.get_realtime_price(ticker)
                    if not price or price <= 0:
                        logger.warning(f"[Skip] {ticker}: Cannot get price, skipping sell order")
                        continue
                
                # 过滤掉交易量过小的订单（只检查股数，不检查金额）
                # 注意：对于不在目标权重中的股票，即使金额 < $100，也应该卖出（清理持仓）
                if abs(current_shares) < 1.0:
                    logger.warning(f"[Skip] {ticker}: Sell order too small ({current_shares:.2f} shares < 1.0)")
                    continue
                
                # 如果金额 < $100，仍然允许卖出（但记录警告）
                if abs(current_shares * price) < 100:
                    logger.warning(f"[Warning] {ticker}: Sell order value is small (${current_shares * price:.2f} < $100), but will still sell to clean up position")
                
                # 检查是否已经在 trades 中（不应该发生，但安全起见）
                if ticker in trades:
                    logger.warning(f"[Warning] {ticker}: Already in trades dict, overwriting with sell order")
                
                trades[ticker] = -abs(current_shares)  # 负数表示卖出
                logger.info(
                    f"[Trade] {ticker}: SELL {abs(current_shares):.2f} shares "
                    f"(not in target portfolio, current={current_shares:.2f}, price=${price:.2f}, value=${current_shares * price:,.2f})"
                )
        
        if not stocks_not_in_target:
            logger.info("  ✅ All current positions are in target portfolio (no stocks to sell)")
        else:
            logger.info(f"  Found {len(stocks_not_in_target)} stocks not in target: {', '.join(stocks_not_in_target)}")

        # 最终安全检查
        if available_cash is not None and total_buy_value > available_cash:
            logger.error(
                f"[Safety] Total buy value (${total_buy_value:,.2f}) exceeds available cash "
                f"(${available_cash:,.2f}). This should not happen!"
            )

        # 验证：确保目标持仓数正确
        target_count = len(target_tickers)
        logger.info(f"[Info] Target portfolio size: {target_count} stocks")
        
        # 详细统计：显示当前持仓和目标持仓的对比
        logger.info(f"[Info] Current positions: {len(self.positions)} stocks")
        if self.positions:
            for ticker, shares in sorted(self.positions.items()):
                weight = target_weights.get(ticker, 0.0)
                price = prices.get(ticker, 0.0)
                target_shares = (total_capital * weight / price) if price > 0 else 0.0
                in_target = ticker in target_tickers
                trade = trades.get(ticker, 0.0)
                status = "✅" if abs(trade) < 1e-6 else ("📈 BUY" if trade > 0 else "📉 SELL")
                logger.info(
                    f"  {status} {ticker}: current={shares:.2f}, target={target_shares:.2f}, "
                    f"trade={trade:.2f}, in_target={in_target}, weight={weight:.4f}"
                )
        else:
            logger.info("  (No current positions)")
        
        # 详细统计
        buy_count = sum(1 for v in trades.values() if v > 0)
        sell_count = sum(1 for v in trades.values() if v < 0)
        logger.info(f"[Info] Total trades: {len(trades)} orders ({buy_count} buys, {sell_count} sells)")
        
        # 详细分析换手情况
        if sell_count > 0:
            sell_tickers = [ticker for ticker, shares in trades.items() if shares < 0]
            logger.info(f"[Turnover] Sell orders for: {', '.join(sell_tickers)}")
            for ticker in sell_tickers:
                shares = trades[ticker]
                current = self.positions.get(ticker, 0.0)
                logger.info(f"  {ticker}: SELL {abs(shares):.2f} shares (current={current:.2f})")
        else:
            logger.info("[Info] No sell orders needed")
            
            # 检查是否有需要换手但没有生成订单的情况
            current_positions = set(self.positions.keys())
            target_positions = target_tickers
            
            # 情况1: 目标权重中的股票，当前持仓 > 目标持仓（应该卖出部分）
            for ticker in target_positions & current_positions:
                if ticker in trades and trades[ticker] > 0:
                    # 已经在买入，不需要检查
                    continue
                weight = target_weights.get(ticker, 0.0)
                if weight > 0:
                    price = prices.get(ticker)
                    if price and price > 0:
                        target_shares = (total_capital * weight) / price
                        current_shares = self.positions.get(ticker, 0.0)
                        if current_shares > target_shares:
                            excess = current_shares - target_shares
                            if abs(excess) >= 1.0 and abs(excess * price) >= 100:
                                logger.warning(f"[Warning] {ticker}: Should SELL {excess:.2f} shares (current={current_shares:.2f} > target={target_shares:.2f}), but no sell order generated!")
            
            # 情况2: 当前持仓中的股票，不在目标权重中（应该全部卖出）
            current_not_in_target = current_positions - target_positions
            if current_not_in_target:
                logger.warning(f"[Warning] Found {len(current_not_in_target)} current positions not in target portfolio:")
                for ticker in current_not_in_target:
                    shares = self.positions[ticker]
                    price = prices.get(ticker, 0.0)
                    value = shares * price if price > 0 else 0.0
                    reason = []
                    if not price or price <= 0:
                        reason.append("no price")
                    if abs(shares) < 1.0:
                        reason.append(f"shares < 1 ({shares:.2f})")
                    if value < 100:
                        reason.append(f"value < $100 (${value:.2f})")
                    if reason:
                        logger.warning(f"  {ticker}: {shares:.2f} shares - skipped because: {', '.join(reason)}")
                    else:
                        logger.warning(f"  {ticker}: {shares:.2f} shares - should be sold but no order generated!")

        return trades

    def place_order(
        self,
        ticker: str,
        trade_shares: float,
        price: float,
    ):
        """下单（限价单或市价单）- 同步版本"""
        try:
            if not self.ib.isConnected():
                logger.error(f"[Error] {ticker}: Not connected to IBKR. Cannot place order.")
                return None

            contract = Stock(ticker, "SMART", "USD")
            # ib_insync 同步 API
            self.ib.qualifyContracts(contract)

            if self.order_type == "LIMIT":
                # 限价单：买入时价格 + offset，卖出时价格 - offset
                limit_price = price * (1 + self.price_offset) if trade_shares > 0 else price * (1 - self.price_offset)
                order = LimitOrder(
                    action="BUY" if trade_shares > 0 else "SELL",
                    totalQuantity=abs(int(trade_shares)),
                    lmtPrice=round(limit_price, 2),
                    outsideRth=True,  # 允许盘前/盘后交易
                )
                logger.info(f"[Order] {ticker}: LIMIT {order.action} {order.totalQuantity} @ ${order.lmtPrice:.2f}")
            elif self.order_type == "MKT":
                # 市价单（风险较高，不推荐）
                order = MarketOrder(
                    action="BUY" if trade_shares > 0 else "SELL",
                    totalQuantity=abs(int(trade_shares)),
                )
                logger.warning(f"[Order] {ticker}: MARKET {order.action} {order.totalQuantity} (RISKY!)")
            else:
                raise ValueError(f"Unsupported order type: {self.order_type}")

            # ib_insync 同步 API：直接调用 placeOrder（返回 Trade 对象）
            trade = self.ib.placeOrder(contract, order)
            self.orders[ticker] = trade
            logger.info(f"[OK] Order placed for {ticker} (orderId={trade.order.orderId})")

            return trade
        except ConnectionError as e:
            logger.error(f"[Error] {ticker}: Connection lost - {e}")
            return None
        except Exception as e:
            logger.error(f"[Error] Failed to place order for {ticker}: {e}")
            return None

    def wait_for_orders(self, timeout: int = 300):
        """等待所有订单成交或超时 - 同步版本"""
        logger.info(f"Waiting for orders to fill (timeout={timeout}s)...")
        start_time = time.time()
        
        # 检查是否有PreSubmitted订单（非交易时间提交的订单）
        pre_submitted_orders = []
        for ticker, trade in self.orders.items():
            if trade.orderStatus.status == "PreSubmitted":
                pre_submitted_orders.append(ticker)
        
        if pre_submitted_orders:
            logger.info(f"[Info] {len(pre_submitted_orders)} orders are in PreSubmitted status (will execute at market open)")
            logger.info(f"  PreSubmitted tickers: {', '.join(pre_submitted_orders)}")
            logger.info("[Info] These orders will be executed automatically when market opens (9:30 AM ET)")
            # 对于PreSubmitted订单，不等待成交，因为它们会在市场开盘时自动执行
            # 只等待其他状态的订单

        while (time.time() - start_time) < timeout:
            all_filled = True
            for ticker, trade in self.orders.items():
                status = trade.orderStatus.status
                filled = trade.orderStatus.filled
                remaining = trade.orderStatus.remaining

                if status == "Filled":
                    logger.info(f"[Filled] {ticker}: {filled} shares filled")
                elif status == "Cancelled":
                    logger.warning(f"[Cancelled] {ticker}: Order was cancelled")
                    # 如果订单被取消，检查是否是PreSubmitted（会被重新提交）
                    if ticker in pre_submitted_orders:
                        logger.info(f"[Info] {ticker}: Order was cancelled but will be resubmitted at market open")
                elif status == "PreSubmitted":
                    # PreSubmitted订单会在市场开盘时自动执行，不在这里等待
                    logger.debug(f"[PreSubmitted] {ticker}: Order will execute at market open (9:30 AM ET)")
                    # 对于PreSubmitted订单，不认为需要等待（它们会在开盘时自动执行）
                    continue
                elif status in ["Submitted", "PendingSubmit"]:
                    logger.debug(f"[Pending] {ticker}: {filled}/{filled + remaining} filled")
                    all_filled = False
                else:
                    logger.debug(f"[Status] {ticker}: {status}")
                    all_filled = False

            # 如果所有非PreSubmitted订单都已成交或取消，退出循环
            if all_filled:
                if pre_submitted_orders:
                    logger.info(f"[OK] All active orders filled. {len(pre_submitted_orders)} PreSubmitted orders will execute at market open.")
                else:
                    logger.info("[OK] All orders filled")
                break

            # 同步 sleep
            self.ib.sleep(2)

        if not all_filled:
            pending_tickers = [t for t, trade in self.orders.items() 
                             if trade.orderStatus.status not in ["Filled", "PreSubmitted", "Cancelled"]]
            if pending_tickers:
                logger.warning(f"[Timeout] Some orders may still be pending after {timeout}s: {', '.join(pending_tickers)}")
            else:
                logger.info("[Info] All orders are either filled or in PreSubmitted status (will execute at market open)")

    def execute_trades(
        self,
        target_weights: pd.Series,
        prices: Dict[str, float],
        total_capital: float,
        available_cash: Optional[float] = None,
    ):
        """执行交易流程 - 同步版本"""
        # 注意：当前持仓已经在 run() 方法中获取，这里不需要重复获取
        # 但如果 self.positions 为空，说明可能有问题，重新获取一次
        if not self.positions:
            logger.warning("[Warning] self.positions is empty, fetching positions again...")
            self.get_current_positions()

        # 2. 计算交易量（传入可用资金用于安全检查）
        trades = self.calculate_trade_shares(target_weights, prices, total_capital, available_cash)

        if not trades:
            logger.info("[Skip] No trades needed (all positions aligned)")
            return

        # 3. 下单
        logger.info(f"Placing {len(trades)} orders...")
        for ticker, trade_shares in trades.items():
            price = prices.get(ticker)
            if price:
                self.place_order(ticker, trade_shares, price)

        # 4. 等待订单成交
        self.wait_for_orders()

    def run(
        self,
        weights_path: Path,
        total_capital: float,
        capital_usage_ratio: float = 0.90,
    ):
        """
        主流程 - 同步版本

        Args:
            weights_path: 权重文件路径
            total_capital: 指定的总资金（如果为 None 或 0，则自动读取账户资金）
            capital_usage_ratio: 资金使用比例（0.90 = 只用90%，留10%缓冲）
        """
        try:
            # 连接
            self.connect()

            # 自动读取账户资金
            buying_power = self.get_account_buying_power()
            if buying_power:
                # 应用资金使用比例（留缓冲）
                available_cash = buying_power * capital_usage_ratio
                logger.info(f"Using {capital_usage_ratio*100:.0f}% of buying power: ${available_cash:,.2f}")

                # 如果用户指定的资金超过可用资金，使用可用资金
                if total_capital > available_cash:
                    logger.warning(
                        f"Specified capital (${total_capital:,.2f}) exceeds available cash "
                        f"(${available_cash:,.2f}). Using available cash instead."
                    )
                    total_capital = available_cash
                elif total_capital <= 0:
                    # 如果用户没有指定或指定为0，使用可用资金
                    total_capital = available_cash
                    logger.info(f"Using auto-detected capital: ${total_capital:,.2f}")
                else:
                    logger.info(f"Using specified capital: ${total_capital:,.2f} (available: ${available_cash:,.2f})")
            else:
                # 无法读取账户资金，使用用户指定的值
                if total_capital <= 0:
                    logger.error("[Error] Cannot auto-detect buying power and no capital specified.")
                    logger.info("Please specify --capital <amount> to continue.")
                    return
                logger.warning(f"Could not auto-detect buying power. Using specified capital: ${total_capital:,.2f}")
                available_cash = None  # 无法获取，不进行资金检查

            # 加载权重
            logger.info(f"Loading weights from {weights_path}...")
            weights_df = pd.read_parquet(weights_path)
            weights_df.index = pd.to_datetime(weights_df.index)
            weights_df = weights_df.sort_index()
            latest_date = weights_df.index[-1]
            target_weights = weights_df.loc[latest_date].fillna(0.0)
            target_weights = target_weights[target_weights > 0]  # 只保留正权重
            target_weights = target_weights / target_weights.sum()  # 归一化
            logger.info(f"[OK] Loaded weights for {latest_date.date()} ({len(target_weights)} tickers)")

            # 关键修复：先获取当前持仓，避免重复买入
            logger.info("\n[Step 1] Fetching current positions...")
            self.get_current_positions()
            
            # 获取实时价格（包括当前持仓中的股票，用于卖出订单）
            tickers = list(set(target_weights.index.tolist() + list(self.positions.keys())))
            prices = self.get_realtime_prices(tickers)

            if not prices:
                logger.error("[Error] No valid prices available. Aborting.")
                return

            # 显示当前持仓和目标持仓的对比
            logger.info(f"\n[Portfolio Comparison]")
            logger.info(f"  Current positions: {len(self.positions)} stocks")
            if self.positions:
                for ticker, shares in sorted(self.positions.items()):
                    logger.info(f"    {ticker}: {shares:.2f} shares")
            else:
                logger.info("    (No current positions)")
            logger.info(f"  Target positions: {len(target_weights[target_weights > 0])} stocks")
            logger.info(f"  Target holdings: {', '.join(sorted(target_weights[target_weights > 0].index))}")
            
            # 检查是否有需要卖出的股票
            current_tickers = set(self.positions.keys())
            target_tickers = set(target_weights[target_weights > 0].index)
            to_sell = current_tickers - target_tickers
            if to_sell:
                logger.info(f"  ⚠️  Stocks to sell (not in target): {', '.join(sorted(to_sell))}")
                for ticker in sorted(to_sell):
                    shares = self.positions[ticker]
                    price = prices.get(ticker, 0.0)
                    value = shares * price if price > 0 else 0.0
                    logger.info(f"    {ticker}: {shares:.2f} shares (${value:,.2f})")
            else:
                logger.info(f"  ✅ No stocks to sell (all current positions are in target portfolio)")
            
            # 检查是否有需要买入的股票
            to_buy = target_tickers - current_tickers
            if to_buy:
                logger.info(f"  📈 Stocks to buy (new positions): {', '.join(sorted(to_buy))}")
            else:
                logger.info(f"  ✅ No new stocks to buy")
            
            # 检查是否有需要调整的股票（在目标中，但持仓数量不对）
            to_adjust = current_tickers & target_tickers
            if to_adjust:
                logger.info(f"  🔄 Stocks to adjust: {', '.join(sorted(to_adjust))}")
            
            # 执行交易（传入可用资金用于安全检查）
            self.execute_trades(target_weights, prices, total_capital, available_cash)

            logger.info("[OK] Trading complete")

        except Exception as e:
            logger.error(f"[Error] Trading failed: {e}", exc_info=True)
        finally:
            self.disconnect()


def main():
    # 使用统一的路径管理
    from src.config.path import OUTPUT_PORTFOLIOS_DIR, get_path
    
    parser = argparse.ArgumentParser(description="IBKR Live Trader (基于 ib_insync，同步版本)")
    default_weights = str(OUTPUT_PORTFOLIOS_DIR / "weights.parquet")
    parser.add_argument("--weights", default=default_weights, help="权重文件路径")
    parser.add_argument("--capital", type=float, default=0.0, help="总资金（0=自动读取账户资金）")
    parser.add_argument("--capital-usage-ratio", type=float, default=0.90, help="资金使用比例（0.90=只用90%%，留10%%缓冲）")
    parser.add_argument("--ib-host", default="127.0.0.1", help="IB Gateway/TWS 主机")
    parser.add_argument("--ib-port", type=int, default=7497, help="端口 (7497=Paper, 7496=Live)")
    parser.add_argument("--client-id", type=int, default=777, help="客户端 ID")
    parser.add_argument("--order-type", default="LIMIT", choices=["LIMIT", "MKT"], help="订单类型")
    parser.add_argument("--price-offset", type=float, default=0.001, help="限价单价格偏移 (0.001=0.1%%)")
    parser.add_argument(
        "--market-data-type",
        default="delayed",
        choices=["real", "delayed", "delayed_frozen"],
        help="行情类型（real/delayed/delayed_frozen）",
    )
    args = parser.parse_args()

    trader = IBKRLiveTrader(
        host=args.ib_host,
        port=args.ib_port,
        client_id=args.client_id,
        order_type=args.order_type,
        price_offset=args.price_offset,
        market_data_type=args.market_data_type,
    )

    # 同步调用（如果用户传入的是相对路径，使用 get_path 解析）
    weights_path = get_path(args.weights, OUTPUT_PORTFOLIOS_DIR) if not Path(args.weights).is_absolute() else Path(args.weights)
    trader.run(weights_path, args.capital, args.capital_usage_ratio)


if __name__ == "__main__":
    main()
