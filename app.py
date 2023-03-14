import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import sqlite3


app = Flask(__name__)

# 设置上传文件保存路径和允许上传的文件类型
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 辅助函数：判断文件是否是允许的类型
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 首页路由，显示一个上传框、预测按钮和输出框
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 处理文件上传请求
        file = request.files['file']
        if file and allowed_file(file.filename):
            # 保存上传的文件
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # 进行模型预测
            # ...
            prediction = "TODO: replace me"

            # 将上传记录保存到数据库
            with sqlite3.connect('database.db') as conn:
                c = conn.cursor()
                upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                c.execute('INSERT INTO uploads (upload_time, image_path, prediction) VALUES (?, ?, ?)',
                          (upload_time, file_path, prediction))
                conn.commit()

            # 显示预测结果
            return render_template('index.html', prediction=prediction, file_path=file_path)

    # 显示首页
    return render_template('index.html')

# 历史记录路由，显示上传历史记录
@app.route('/history')
def history():
    # 查询所有的上传记录
    with sqlite3.connect('database.db') as conn:
        c = conn.cursor()
        c.execute('SELECT * FROM uploads ORDER BY id DESC')
        rows = c.fetchall()

    # 显示历史上传记录
    return render_template('history.html', rows=rows)

# 删除指定ID的上传记录以及对应的图片文件
@app.route('/delete/<int:id>')
def delete(id):
    # 查询指定ID的上传记录
    with sqlite3.connect('database.db') as conn:
        c = conn.cursor()
        c.execute('SELECT image_path FROM uploads WHERE id=?', (id,))
        row = c.fetchone()

        if row is not None:
            # 删除对应的图片文件
            os.remove(row[0])

            # 删除上传记录
            c.execute('DELETE FROM uploads WHERE id=?', (id,))
            conn.commit()

    # 返回历史记录页面
    return redirect(url_for('history'))

if __name__ == '__main__':
    app.run(debug=True)
