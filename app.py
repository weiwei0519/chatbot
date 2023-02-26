# coding=UTF-8
# Web应用的主体，实现页面交互，服务调用和检测
# 

'''
@File: app.py
@Author: Wei Wei
@Time: 2022/7/30 17:13
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''
from functools import wraps
from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import execute
import time
import threading
import jieba
import json


# 定义心跳检测函数
def heartbeat():
    print(time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()

# 允许跨域
def allow_cross_domain(fun):
    @wraps(fun)
    def wrapper_fun(*args, **kwargs):
        rst = make_response(fun(*args, **kwargs))
        rst.headers['Access-Control-Allow-Origin'] = '*'
        rst.headers['Access-Control-Allow-Methods'] = 'GET,POST'
        allow_headers = "Referer,Accept,Origin,User-Agent"
        rst.headers['Access-Control-Allow-Headers'] = allow_headers
        return rst
    return wrapper_fun


timer = threading.Timer(60, heartbeat)
timer.start()
app = Flask(__name__, static_url_path="/static")
CORS(app, supports_credentials=True)  # 解决flask跨域问题


@app.route('/message', methods=['POST', 'GET'])
@allow_cross_domain
# """定义应答函数，用于获取输入信息并返回相应的答案"""
def reply():
    # 从请求中获取参数信息
    try:
        req_msg = request.form['msg']
    except Exception:
        req_data = request.get_data()
        req_msg = json.loads(req_data)['msg']
    # 将语句使用结巴分词进行分词
    req_msg = " ".join(jieba.cut(req_msg))
    # 调用decode_line对生成回答信息
    res_msg = execute.predict(req_msg)
    # 将unk值的词用微笑符号袋贴
    res_msg = res_msg.replace('_UNK', '^_^')
    res_msg = res_msg.strip()
    res_msg = res_msg.replace(' ', '')
    # 如果接受到的内容为空，则给出相应的回复
    if res_msg == ' ':
        res_msg = '请与我聊聊天吧'
    # jsonify:是用于处理序列化json数据的函数，就是将数据组装成json格式返回
    # http://flask.pocoo.org/docs/0.12/api/#module-flask.json
    return jsonify({'text': res_msg})


@app.route("/")
def index():
    return render_template("index.html")


# 启动APP
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8808)
