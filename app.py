import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import sqlite3
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn

app = Flask(__name__)

# 设置上传文件保存路径和允许上传的文件类型
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 加载PyTorch模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 第一层卷积，输入通道为1，输出通道为6，卷积核大小为5，边缘填充为2
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        # 第一层池化，使用最大池化，池化核大小为2
        self.pool1 = nn.MaxPool2d(2)
        # 第二层卷积，输入通道为6，输出通道为16，卷积核大小为5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 第二层池化，使用最大池化，池化核大小为2
        self.pool2 = nn.MaxPool2d(2)
        # 第一层全连接，输入特征维度为16*5*5，输出特征维度为120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 第二层全连接，输入特征维度为120，输出特征维度为84
        self.fc2 = nn.Linear(120, 84)
        # 第三层全连接，输入特征维度为84，输出特征维度为10（分类数）
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 前向传播过程
        x = self.pool1(torch.relu(self.conv1(x)))  # 对第一层卷积的输出应用relu激活函数和池化操作
        x = self.pool2(torch.relu(self.conv2(x)))  # 对第二层卷积的输出应用relu激活函数和池化操作
        x = x.view(-1, 16 * 5 * 5)  # 将二维特征图展平为一维向量
        x = torch.relu(self.fc1(x))  # 对第一层全连接的输出应用relu激活函数
        x = torch.relu(self.fc2(x))  # 对第二层全连接的输出应用relu激活函数
        x = self.fc3(x)  # 对第三层全连接的输出不应用激活函数（最后一层）
        return x  # 返回模型的预测结果


model = LeNet5()
model.load_state_dict(torch.load('lenet5.pth'))

# 定义分类标签
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# 辅助函数：判断文件是否是允许的类型
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# 辅助函数：对上传的图像进行预处理，并送入模型进行预测
def predict_image(image_path):
    # 加载图像并预处理
    img = Image.open(image_path).convert('L')
    transform = transforms.Compose([

        transforms.ToTensor(),
    ])
    img_tensor = transform(img)
    img_batch = img_tensor.unsqueeze(0)

    # 使用模型进行预测
    model.eval()
    logits = model(img_batch)
    probs = torch.softmax(logits, dim=1)
    top_probs, top_labels = probs.topk(5, dim=1)

    # 构造预测结果字符串
    pred_str = ''
    for i in range(top_probs.size(1)):
        prob = top_probs[0][i].item()
        label_idx = top_labels[0][i].item()
        label = labels[label_idx]
        pred_str += f'{label}: {prob:.3f}\n'
    return pred_str.strip()


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
            prediction = predict_image(file_path)

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
