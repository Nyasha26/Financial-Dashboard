from data import get_series

if __name__ == "__main__":
    df = get_series("DGS10")
    print(f"Rows: {len(df)}")
    print(df.tail(5))
