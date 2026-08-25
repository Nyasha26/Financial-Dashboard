# Nyasha Mugabe Dashboard

A Streamlit dashboard for FRED (Federal Reserve Economic Data) series, with a
single-series explorer and a market-regime analysis of S&P 500 returns
across yield-curve, dollar, and Fed-funds regimes.

## Run locally

```
pip install -r requirements.txt
```

Create a `.env` file in this folder with:

```
FRED_API_KEY=your_key_here
```

Then:

```
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (get a permanent link)

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click "New app".
3. Pick this repo, branch `main`, and set the main file to `app.py`.
4. Before (or after) deploying, open the app's **Settings -> Secrets** and
   add:
   ```
   FRED_API_KEY = "your_key_here"
   ```
   `config.py` reads it the same way it reads a local `.env` file, so no
   code changes are needed.
5. Deploy. You'll get a permanent URL like
   `https://<app-name>-<random>.streamlit.app` that stays live as long as
   the app isn't put to sleep from inactivity (Streamlit Cloud spins down
   idle free-tier apps after a period of no traffic, but a visit wakes it
   back up).
