import os
import json
import logging
import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import get_as_dataframe
from sklearn.preprocessing import StandardScaler

LOCAL_KEY_FILE = 'service_account_key.json'

logger = logging.getLogger("data_processor")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('log.txt')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_data():
    logger.info("Rozpoczęcie pobierania danych z Google Sheets")

    credentials_json = os.environ.get("GCP_CREDENTIALS")
    SHEET_ID = os.environ.get("SHEET_ID")

    if credentials_json is None or SHEET_ID is None:
        raise ValueError("Brak sekretów GCP_CREDENTIALS lub SHEET_ID.")
    
    scope = ["https://spreadsheets.google.com/feeds",  "https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        json.loads(credentials_json.strip()),
        scopes=scope
    )
    gc = gspread.authorize(credentials)

    spreadsheet = gc.open_by_key(SHEET_ID)
    worksheet = spreadsheet.sheet1

    df = get_as_dataframe(worksheet, header=0)
    
    initial_rows = len(df)
    initial_cells = df.size
    
    logger.info(f"Pomyślnie pobrano dane. Początkowa liczba wierszy: {initial_rows}")
    return df

def process_data(df):
    logger.info("Rozpoczęcie czyszczenia i standaryzacji danych")

    NUMERIC_COLS = ['Wiek', 'Średnie Zarobki']
    TIME_COLS = ['Czas Początkowy Podróży', 'Czas Końcowy Podróży']
    CATEGORICAL_COLS = ['Płeć', 'Wykształcenie', 'Cel Podróży']

    initial_rows = len(df)
    initial_cells = df.size
    rows_deleted_count = 0
    cells_imputed_count = 0

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    THRESHOLD_COUNT = len(df.columns) - 2

    rows_deleted_count = initial_rows - len(df.dropna(thresh=THRESHOLD_COUNT))
    df.dropna(thresh=THRESHOLD_COUNT, inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(f"Usunięto {rows_deleted_count} wierszy o zbyt dużej liczbie braków.")

    for col in NUMERIC_COLS:
        nan_count = df[col].isnull().sum()
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        cells_imputed_count += nan_count

    for col in CATEGORICAL_COLS:
        nan_count = df[col].isnull().sum()
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        cells_imputed_count += nan_count

    logger.info(f"Uzupełniono braki w {cells_imputed_count} komórkach (medianą lub modą).")

    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    cells_standardized_count = len(df) * len(NUMERIC_COLS)
    
    logger.info(f"Standaryzacja Z-Score dla kolumn {NUMERIC_COLS} zakończona.")

    perc_deleted_rows = (rows_deleted_count / initial_rows) * 100 if initial_rows > 0 else 0
    perc_changed_cells = (cells_imputed_count / initial_cells) * 100 if initial_cells > 0 else 0
    

    total_changed_cells = cells_imputed_count + cells_standardized_count
    perc_deleted_rows = (rows_deleted_count / initial_rows) * 100 if initial_rows > 0 else 0
    perc_changed_cells = (total_changed_cells / initial_cells) * 100 if initial_cells > 0 else 0
    
    # Zapis raportu do pliku
    with open('report.txt', 'w') as f:
        f.write("Raport z Czyszczenia i Standaryzacji Danych (Lab2)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Początkowa liczba wierszy: {initial_rows}\n")
        f.write(f"Wiersze usunięte (nadmiar braków): {rows_deleted_count} ({perc_deleted_rows:.2f}%)\n")
        f.write(f"Komórki uzupełnione brakami: {cells_imputed_count}\n")
        f.write(f"Procent wszystkich komórek, które zostały zmienione: {perc_changed_cells:.2f}%\n")
        f.write(f"Końcowa liczba wierszy: {len(df)}\n")
    
    logger.info(f"Raport zapisany do report.txt. Zwracam przetworzony DataFrame.")

if __name__ == "__main__":
    df = get_data()
    process_data(df)


