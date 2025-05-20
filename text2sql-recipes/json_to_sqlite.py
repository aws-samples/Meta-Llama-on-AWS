import sqlite3
import json
from typing import Optional
from pathlib import Path

class JsonToSqlite:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.conn = None
        self.cursor = None

    def connect(self):
        self.conn = sqlite3.connect(self.db_file)
        self.cursor = self.conn.cursor()

    def close(self):
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def create_tables(self):
        # Create airplanes table
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS airplanes (
            Airplane_id INTEGER PRIMARY KEY,
            Producer TEXT,
            Type TEXT
        )''')

        # Create flights table
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS flights (
            Flight_number TEXT PRIMARY KEY,
            Arrival_time DATETIME,
            Arrival_date DATE,
            Departure_time DATETIME,
            Departure_date DATE,
            Destination TEXT,
            Airplane_id INTEGER,
            FOREIGN KEY (Airplane_id) REFERENCES airplanes(Airplane_id)
        )''')

    def import_flights(self, json_data: list):
        for flight in json_data:
            self.cursor.execute('''
                INSERT OR REPLACE INTO flights (
                    Flight_number,
                    Arrival_time,
                    Arrival_date,
                    Departure_time,
                    Departure_date,
                    Destination,
                    Airplane_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                flight['Flight_number'],
                flight['Arrival_time'],
                flight['Arrival_date'],
                flight['Departure_time'],
                flight['Departure_date'],
                flight['Destination'],
                flight['Airplane_id']
            ))

    def import_airplanes(self, json_data: list):
        for airplane in json_data:
            self.cursor.execute('''
                INSERT OR REPLACE INTO airplanes (
                    Airplane_id,
                    Producer,
                    Type
                ) VALUES (?, ?, ?)
            ''', (
                airplane['Airplane_id'],
                airplane['Producer'],
                airplane['Type']
            ))

    def process_json_files(self, flights_json: str, airplanes_json: str) -> bool:
        try:
            self.connect()
            self.create_tables()

            with open(airplanes_json, 'r') as file:
                airplanes_data = json.load(file)
                self.import_airplanes(airplanes_data)

            with open(flights_json, 'r') as file:
                flights_data = json.load(file)
                self.import_flights(flights_data)

            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return False
        finally:
            self.close()