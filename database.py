import sqlite3

# 连接到数据库文件
conn = sqlite3.connect('database.db')

# 创建一个名为"uploads"的表格，包含id、上传时间、图片路径和预测结果四个字段
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS uploads
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              upload_time TEXT,
              image_path TEXT,
              prediction TEXT)''')
conn.commit()
