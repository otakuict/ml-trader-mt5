import time
from datetime import datetime, timedelta

import MetaTrader5 as mt5

from settings import LOT_SIZE, SYMBOL
DEVIATION = 20
MAGIC = 51002


def _normalize_volume(volume: float, symbol_info) -> float:
    min_vol = getattr(symbol_info, "volume_min", volume)
    max_vol = getattr(symbol_info, "volume_max", volume)
    step = getattr(symbol_info, "volume_step", 0.01)
    vol = max(volume, min_vol)
    vol = min(vol, max_vol)
    if step:
        vol = round(vol / step) * step
        if vol < min_vol:
            vol = min_vol
    return float(vol)


def _place_market(
    symbol: str,
    volume: float,
    order_type: int,
    position: int | None = None,
) -> mt5.OrderSendResult | None:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("No tick data.")
        return None
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "ml-trader-test",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    if position is not None:
        request["position"] = position
    return mt5.order_send(request)


def main() -> int:
    if not mt5.initialize():
        print("MT5 initialize failed:", mt5.last_error())
        return 1
    try:
        account = mt5.account_info()
        if not account:
            print("Account info unavailable.")
            return 1
        if getattr(account, "trade_mode", None) != mt5.ACCOUNT_TRADE_MODE_DEMO:
            print("Not a demo account. Aborting test order.")
            return 1

        symbol_info = mt5.symbol_info(SYMBOL)
        if symbol_info is None:
            print(f"Symbol not found: {SYMBOL}")
            return 1
        if not symbol_info.visible:
            if not mt5.symbol_select(SYMBOL, True):
                print(f"Failed to select symbol: {SYMBOL}")
                return 1

        volume = _normalize_volume(LOT_SIZE, symbol_info)
        print(f"Placing test BUY {volume} {SYMBOL} ...")
        result = _place_market(SYMBOL, volume, mt5.ORDER_TYPE_BUY)
        if result is None:
            print("Order send failed.")
            return 1
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed: retcode={result.retcode} comment={result.comment}")
            return 1

        ticket = result.order
        print(f"Order placed. Ticket: {ticket}")

        time.sleep(2)
        positions = mt5.positions_get(ticket=ticket)
        if positions:
            pos = positions[0]
            close_type = (
                mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            )
            print("Closing test position ...")
            close = _place_market(SYMBOL, pos.volume, close_type, position=pos.ticket)
            if close and close.retcode == mt5.TRADE_RETCODE_DONE:
                print("Position closed.")
            else:
                print("Close failed.")

        since = datetime.now() - timedelta(days=1)
        deals = mt5.history_deals_get(since, datetime.now())
        if deals:
            print("Recent deals (last 24h):")
            for d in deals[-5:]:
                print(
                    f"deal={d.ticket} order={d.order} type={d.type} volume={d.volume} price={d.price}"
                )
        else:
            print("No deals found in last 24h.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
