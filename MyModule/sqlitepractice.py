import sqlite3
import pandas as pd


class Database():
    
    def __init__(self):

        self.table = 'datatable'
        self.create_sql_table = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            description TEXT NOT NULL,
                            price INTEGER,
                            quantity INTEGER);
                            """
        self.create_bar_qr_table = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            type TEXT NOT NULL,
                            data TEXT NOT NULL);
                            """
        self.insert_sql = f"INSERT INTO {self.table} (name, description, price, quantity) VALUES ( ? , ? , ? , ? )"
        self.insert_barcode_qr_sql = f"INSERT INTO {self.table} (type, data) VALUES ( ? , ? )"
        self.select_sql = f"SELECT * FROM {self.table}"
        self.delete_sql = f"DELETE FROM {self.table} WHERE name = ?"
        self.update_sql = f"UPDATE {self.table} SET price = ? WHERE name = ?"
        self.inout_sql = f"UPDATE {self.table} SET quantity = ? WHERE description = ?"
        self.remove_sql = f"DROP TABLE {self.table};"
        self.remove_sql = f"DROP TABLE {self.table};"

        self.name = f'{input('DB파일명을 입력하세요. : ')}.db'

    def create_table(self):
        db = self.name
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute(self.create_bar_qr_table)
        conn.commit()
        conn.close()

    def insert_bar_qr(self,params=()):
        db = self.name
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute(self.insert_barcode_qr_sql,params)
        conn.commit()
        conn.close()

    def reset_table(self):
        db = self.name
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute(self.remove_sql)
        conn.commit()
        conn.close()

    def compare(self,_data):

        db = self.name
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {self.table}")
        rows = cursor.fetchall()
     
        for row in rows:
            if _data == row[2]:
                return False
            else:
                continue
        
        return True

            
    # def execute_query(params=()):
    #     db = f'{input('DB파일명을 입력하세요. : ')}.db'
    #     conn = sqlite3.connect(db)
    #     cursor = conn.cursor()
    #     if isinstance(params,list):
    #         cursor.executemany(query,params)
    #     else:
    #         cursor.execute(query,params)
    #     conn.commit()
    #     conn.close()

    # def inout(self,params):
    #     db = f'{input('DB파일명을 입력하세요. : ')}.db'
    #     conn = sqlite3.connect(db)
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT description,quantity FROM my_table")
    #     rows = cursor.fetchall()
    #     params = list(params)
    #     for row in rows:
    #         if row[0] == params[1]:
    #             params[0] += row[1]
            
    #     cursor.execute(self.inout_sql,params)

    #     conn.commit()
    #     conn.close()

    # def fetch_all_data():
    #     db = f'{input('DB파일명을 입력하세요. : ')}.db'
    #     conn = sqlite3.connect(db)
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT * FROM my_table")
    #     rows = cursor.fetchall()
    #     # print(rows)
    #     for row in rows:
    #         print(f'id[{row[0]}] / name : {row[1]} / description : {row[2]} / price : {row[3]}] / quantity : {row[4]}')
        
    #     conn.commit()
    #     conn.close()