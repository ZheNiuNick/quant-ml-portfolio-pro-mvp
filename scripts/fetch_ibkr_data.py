#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IBKR 数据获取脚本
定期获取 IBKR 的真实持仓、交易记录和收益情况，保存到文件

支持两种方式：
1. Flex Query API（推荐）- 获取所有历史数据
2. ib_insync API（备用）- 获取当前会话数据

使用方法：
1. 使用 Flex Query API（推荐）：
   export IBKR_FLEX_TOKEN="your_flex_token"
   export IBKR_FLEX_QUERY_ID="your_query_id"
   python scripts/fetch_ibkr_data.py

2. 使用 ib_insync API（备用）：
   python scripts/fetch_ibkr_data.py --use-ib-insync
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json
import pandas as pd
import argparse

# 使用统一的路径管理
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.path import OUTPUT_IBKR_DATA_DIR
from src.ibkr_live_trader import IBKRLiveTrader

# 输出目录
OUTPUT_DIR = OUTPUT_IBKR_DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IBKR 配置（从环境变量或默认值）
import os
IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
IBKR_PORT = int(os.getenv('IBKR_PORT', '7497'))
IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '777'))

# Flex Query 配置（从环境变量）
IBKR_FLEX_TOKEN = os.getenv('IBKR_FLEX_TOKEN', '')
IBKR_FLEX_QUERY_ID = os.getenv('IBKR_FLEX_QUERY_ID', '')


def fetch_positions(trader):
    """获取持仓数据"""
    try:
        # get_current_positions() 返回 Dict[str, float] (ticker -> shares)
        positions_dict = trader.get_current_positions()
        positions_data = []
        
        # 从 IBKR 获取完整的持仓信息（包括 avgCost）
        ib_positions = {}
        for pos in trader.ib.positions():
            if pos.contract.secType == "STK":
                ticker = pos.contract.symbol
                ib_positions[ticker] = {
                    "shares": float(pos.position),
                    "avg_cost": float(pos.avgCost) if hasattr(pos, 'avgCost') else 0.0
                }
        
        for ticker, shares in positions_dict.items():
            current_price = trader.get_realtime_price(ticker)
            avg_cost = ib_positions.get(ticker, {}).get("avg_cost", 0.0)
            market_value = shares * current_price if current_price else 0.0
            unrealized_pnl = (current_price - avg_cost) * shares if current_price and avg_cost > 0 else 0.0
            
            positions_data.append({
                "symbol": ticker,
                "shares": float(shares),
                "avg_cost": float(avg_cost),
                "current_price": float(current_price) if current_price else 0.0,
                "market_value": float(market_value),
                "unrealized_pnl": float(unrealized_pnl)
            })
        
        return positions_data
    except Exception as e:
        print(f"[Error] Failed to fetch positions: {e}")
        import traceback
        traceback.print_exc()
        return []


def fetch_trades(trader):
    """获取交易记录"""
    try:
        # 使用 IBKRLiveTrader 的内建交易收集器
        # 该方法会自动从 IBKR TWS 获取所有历史交易记录（不仅仅是当前 session）
        trades = trader.get_trades()
        print(f"[OK] 获取到 {len(trades)} 条交易记录")
        return trades
    except Exception as e:
        print(f"[Error] Failed to fetch trades: {e}")
        import traceback
        traceback.print_exc()
        return []


def fetch_pnl(trader):
    """获取收益数据"""
    try:
        # 检测账户类型（Paper Trading vs Real Money）
        account_type = "Unknown"
        account_id = None
        
        # 通过端口判断（7497=Paper, 7496=Live）
        if IBKR_PORT == 7497:
            account_type = "Paper Trading"
        elif IBKR_PORT == 7496:
            account_type = "Real Money"
        else:
            account_type = f"Port {IBKR_PORT}"
        
        # 尝试从账户信息中获取账户ID
        try:
            account_summary = trader.ib.accountSummary()
            if account_summary:
                account_id = account_summary[0].account
                # 检查账户ID是否包含paper/demo等关键词
                if account_id and ('paper' in account_id.lower() or 'demo' in account_id.lower() or 'sim' in account_id.lower()):
                    account_type = "Paper Trading"
        except:
            pass
        
        account_values = trader.ib.accountValues()
        
        pnl_data = {
            "account_type": account_type,
            "account_id": account_id,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "net_liquidation": 0.0,
            "buying_power": 0.0,
            "cash": 0.0,
            "positions_pnl": [],
            "daily_pnl": []
        }
        
        for av in account_values:
            tag = av.tag
            value_str = av.value if av.value else None
            
            # 跳过非数字值（如账户ID等）
            if value_str is None:
                continue
            
            try:
                value = float(value_str)
            except (ValueError, TypeError):
                # 如果无法转换为数字，跳过（可能是账户ID等字符串）
                continue
            
            if tag == "RealizedPnL":
                pnl_data["realized_pnl"] = value
            elif tag == "UnrealizedPnL":
                pnl_data["unrealized_pnl"] = value
            elif tag == "TotalPnL":
                pnl_data["total_pnl"] = value
            elif tag == "NetLiquidation":
                pnl_data["net_liquidation"] = value
            elif tag == "BuyingPower":
                pnl_data["buying_power"] = value
            elif tag == "CashBalance":
                pnl_data["cash"] = value
        
        # 计算总盈亏和盈亏百分比
        total_profit_loss = pnl_data["realized_pnl"] + pnl_data["unrealized_pnl"]
        profit_loss_percent = (total_profit_loss / pnl_data["net_liquidation"] * 100) if pnl_data["net_liquidation"] > 0 else 0.0
        pnl_data["total_profit_loss"] = total_profit_loss
        pnl_data["profit_loss_percent"] = profit_loss_percent
        
        # 获取持仓的未实现盈亏
        for pos in trader.ib.positions():
            if pos.contract.secType == "STK":
                ticker = pos.contract.symbol
                shares = pos.position
                avg_cost = pos.avgCost if hasattr(pos, 'avgCost') else 0.0
                current_price = trader.get_realtime_price(ticker)
                
                if current_price and current_price > 0 and avg_cost > 0:
                    pnl = (current_price - avg_cost) * shares
                    pnl_data["positions_pnl"].append({
                        "symbol": ticker,
                        "shares": float(shares),
                        "avg_cost": float(avg_cost),
                        "current_price": float(current_price),
                        "unrealized_pnl": float(pnl),
                        "value": float(abs(shares) * current_price)
                    })
        
        return pnl_data
    except Exception as e:
        print(f"[Error] Failed to fetch PnL: {e}")
        return {
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
            "net_liquidation": 0.0,
            "buying_power": 0.0,
            "cash": 0.0,
            "positions_pnl": [],
            "daily_pnl": []
        }


def fetch_data_via_flex_query(start_date=None, end_date=None):
    """使用 Flex Query API 获取所有历史数据"""
    try:
        from src.ibkr_flex_query import IBKRFlexQuery
        
        if not IBKR_FLEX_TOKEN or not IBKR_FLEX_QUERY_ID:
            print("[Error] Flex Query Token 或 Query ID 未设置")
            print("请设置环境变量：")
            print("  export IBKR_FLEX_TOKEN='your_token'")
            print("  export IBKR_FLEX_QUERY_ID='your_query_id'")
            print("\n[Note] Paper Account 不支持 Flex Query API，请使用 --use-ib-insync")
            return None, None, None
        
        print("\n[Info] 使用 Flex Query API 获取历史数据...")
        print("[Note] 如果使用 Paper Account，Flex Query 不可用，将自动回退到 ib_insync")
        flex = IBKRFlexQuery(token=IBKR_FLEX_TOKEN, query_id=IBKR_FLEX_QUERY_ID)
        
        # 获取所有历史执行记录
        print("\n[Info] 获取所有历史交易记录...")
        trades = flex.get_executions(start_date=start_date, end_date=end_date)
        print(f"[OK] 获取到 {len(trades)} 条历史交易记录")
        
        # 获取佣金记录
        print("\n[Info] 获取佣金记录...")
        commissions = flex.get_commissions(start_date=start_date, end_date=end_date)
        print(f"[OK] 获取到 {len(commissions)} 条佣金记录")
        
        # 合并佣金到交易记录
        if commissions:
            comm_dict = {c.get('execId', ''): c for c in commissions}
            for trade in trades:
                exec_id = trade.get('execId', '')
                if exec_id in comm_dict:
                    trade['commission'] = float(comm_dict[exec_id].get('commission', 0))
        
        # 获取历史持仓（最新日期）
        print("\n[Info] 获取历史持仓...")
        positions = flex.get_positions(date=end_date)
        print(f"[OK] 获取到 {len(positions)} 个历史持仓")
        
        # 标准化持仓数据
        positions_data = []
        for pos in positions:
            positions_data.append({
                "symbol": pos.get('symbol', ''),
                "shares": float(pos.get('position', 0)),
                "avg_cost": float(pos.get('averageCost', 0)),
                "current_price": float(pos.get('markPrice', 0)),
                "market_value": float(pos.get('marketValue', 0)),
                "unrealized_pnl": float(pos.get('unrealizedPNL', 0)),
            })
        
        # 获取已实现盈亏
        print("\n[Info] 获取已实现盈亏...")
        realized_pnl = flex.get_realized_pnl(start_date=start_date, end_date=end_date)
        print(f"[OK] 获取到 {len(realized_pnl)} 条已实现盈亏记录")
        
        # 计算总已实现盈亏
        total_realized_pnl = sum(float(p.get('realizedPNL', 0)) for p in realized_pnl)
        
        # 获取现金流
        print("\n[Info] 获取现金流记录...")
        cashflows = flex.get_cashflows(start_date=start_date, end_date=end_date)
        print(f"[OK] 获取到 {len(cashflows)} 条现金流记录")
        
        # 计算总现金流
        total_cashflow = sum(float(cf.get('amount', 0)) for cf in cashflows)
        
        # 构建 PnL 数据
        pnl_data = {
            "realized_pnl": total_realized_pnl,
            "unrealized_pnl": sum(p.get("unrealized_pnl", 0) for p in positions_data),
            "total_pnl": total_realized_pnl + sum(p.get("unrealized_pnl", 0) for p in positions_data),
            "net_liquidation": sum(p.get("market_value", 0) for p in positions_data),
            "buying_power": 0.0,  # Flex Query 可能不提供
            "cash": total_cashflow,
            "positions_pnl": positions_data,
            "daily_pnl": realized_pnl,
        }
        
        return positions_data, trades, pnl_data
        
    except ImportError as e:
        print(f"[Error] 无法导入 Flex Query 模块: {e}")
        return None, None, None
    except Exception as e:
        print(f"[Error] Flex Query 获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='IBKR 数据获取脚本')
    parser.add_argument('--use-ib-insync', action='store_true', 
                       help='强制使用 ib_insync API（而不是 Flex Query）')
    parser.add_argument('--start-date', type=str, default=None,
                       help='开始日期 (YYYY-MM-DD)，仅用于 Flex Query')
    parser.add_argument('--end-date', type=str, default=None,
                       help='结束日期 (YYYY-MM-DD)，仅用于 Flex Query，默认为今天')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("IBKR 数据获取脚本")
    print("=" * 60)
    
    # 优先使用 Flex Query API（如果可用且未强制使用 ib_insync）
    # 注意：Paper Account 不支持 Flex Query，会自动回退到 ib_insync
    use_flex = not args.use_ib_insync and IBKR_FLEX_TOKEN and IBKR_FLEX_QUERY_ID
    
    if use_flex:
        print("\n[Info] 尝试使用 Flex Query API（获取所有历史数据）")
        print("[Note] 如果使用 Paper Account，Flex Query 不可用，将自动回退到 ib_insync")
        positions, trades, pnl = fetch_data_via_flex_query(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if positions is None:
            print("\n[Warn] Flex Query 失败，回退到 ib_insync API...")
            print("[Note] 如果使用 Paper Account，这是正常现象（Paper Account 不支持 Flex Query）")
            use_flex = False
    
    # 如果 Flex Query 不可用或失败，使用 ib_insync
    if not use_flex:
        print("\n[Info] 使用 ib_insync API（获取当前会话数据）")
        trader = None
        try:
            # 连接 IBKR
            print(f"\n[Info] 连接 IBKR ({IBKR_HOST}:{IBKR_PORT})...")
            trader = IBKRLiveTrader(
                host=IBKR_HOST,
                port=IBKR_PORT,
                client_id=IBKR_CLIENT_ID
            )
            trader.connect()
            print("[OK] IBKR 连接成功")
            
            # 等待连接稳定
            trader.ib.sleep(2)
            
            # 获取数据
            print("\n[Info] 获取持仓数据...")
            positions = fetch_positions(trader)
            print(f"[OK] 获取到 {len(positions)} 个持仓")
            
            print("\n[Info] 获取交易记录...")
            trades = fetch_trades(trader)
            print(f"[OK] 获取到 {len(trades)} 条交易记录")
            
            print("\n[Info] 获取收益数据...")
            pnl = fetch_pnl(trader)
            print(f"[OK] 收益数据获取完成")
            
        except Exception as e:
            print(f"\n[Error] 获取数据失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            if trader:
                try:
                    trader.disconnect()
                except:
                    pass
    
    # 保存数据（累积模式：合并新旧数据）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存持仓（覆盖，因为持仓是当前状态）
    positions_file = OUTPUT_DIR / "positions.json"
    with open(positions_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "positions": positions or [],
            "total_stocks": len(positions) if positions else 0,
            "total_value": sum(p.get("market_value", 0.0) for p in positions) if positions else 0.0,
            "source": "flex_query" if use_flex else "ib_insync"
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 持仓数据已保存到: {positions_file}")
    
    # 保存交易记录（累积模式：合并新旧交易记录）
    trades_file = OUTPUT_DIR / "trades.json"
    existing_trades = []
    existing_trade_ids = set()
    
    # 读取现有交易记录
    if trades_file.exists():
        try:
            with open(trades_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, dict) and "trades" in existing_data:
                    existing_trades = existing_data.get("trades", [])
                elif isinstance(existing_data, list):
                    existing_trades = existing_data
            
            # 收集现有交易的唯一标识（使用 time + symbol + side + quantity）
            for trade in existing_trades:
                trade_id = f"{trade.get('time', '')}_{trade.get('symbol', '')}_{trade.get('side', '')}_{trade.get('quantity', 0)}"
                existing_trade_ids.add(trade_id)
        except Exception as e:
            print(f"[Warn] 读取现有交易记录失败: {e}")
    
    # 合并新旧交易记录（去重）
    all_trades = existing_trades.copy()
    new_trades_count = 0
    if trades:
        for trade in trades:
            trade_id = f"{trade.get('time', '')}_{trade.get('symbol', '')}_{trade.get('side', '')}_{trade.get('quantity', 0)}"
            if trade_id not in existing_trade_ids:
                all_trades.append(trade)
                existing_trade_ids.add(trade_id)
                new_trades_count += 1
    
    # 按时间排序（最新的在前）
    all_trades.sort(key=lambda x: x.get('time', ''), reverse=True)
    
    with open(trades_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "trades": all_trades,
            "total_trades": len(all_trades),
            "new_trades_today": new_trades_count,
            "source": "flex_query" if use_flex else "ib_insync"
        }, f, indent=2, ensure_ascii=False)
    print(f"[OK] 交易记录已保存到: {trades_file} (总计: {len(all_trades)} 条, 今日新增: {new_trades_count} 条)")
    
    # 保存收益数据（覆盖，因为收益是当前状态）
    pnl_file = OUTPUT_DIR / "pnl.json"
    pnl_data = {
        "timestamp": timestamp,
        "source": "flex_query" if use_flex else "ib_insync",
        **(pnl or {})
    }
    with open(pnl_file, 'w', encoding='utf-8') as f:
        json.dump(pnl_data, f, indent=2, ensure_ascii=False)
    print(f"[OK] 收益数据已保存到: {pnl_file}")
    
    print("\n" + "=" * 60)
    print("✅ 所有数据获取完成！")
    print(f"数据来源: {'Flex Query API' if use_flex else 'ib_insync API'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

