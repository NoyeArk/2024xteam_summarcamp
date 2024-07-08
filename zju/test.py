# -*- encoding:utf-8 -*-
import base64
import hashlib
import hmac
import json
import time
from datetime import datetime
from time import mktime
from urllib.parse import urlencode, urlparse
from wsgiref.handlers import format_date_time
from urllib import parse

import requests

appid = "698c8614"  # 填写控制台中获取的 APPID 信息
apiSecret = "NDJhOTIzODljY2ZiMDk4OTg5MmZjYzI4"  # 填写控制台中获取的 APISecret 信息
apiKey = "2fb3adb3314e207f14c4fce53e82b67a"  # 填写控制台中获取的 APIKey 信息

imagedata = open("data/1.jpg", 'rb').read()
image = str(base64.b64encode(imagedata), 'utf-8')

# 请求地址
create_host_url = "https://cn-huadong-1.xf-yun.com/v1/private/s3fd61810/create"
query_host_url = "https://cn-huadong-1.xf-yun.com/v1/private/s3fd61810/query"


def build_auth_request_url(request_url):
    url_result = parse.urlparse(request_url)
    date = "Thu, 09 May 2024 02:09:13 GMT"  # format_date_time(mktime(datetime.now().timetuple()))
    print(date)
    method = "POST"
    signature_origin = "host: {}\ndate: {}\n{} {} HTTP/1.1".format(url_result.hostname, date, method, url_result.path)
    signature_sha = hmac.new(apiSecret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()
    signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
        apiKey, "hmac-sha256", "host date request-line", signature_sha)
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    values = {
        "host": url_result.hostname,
        "date": date,
        "authorization": authorization
    }
    return request_url + "?" + urlencode(values)


def create_url(url):
    host = urlparse(url).netloc
    path = urlparse(url).path

    # 生成RFC1123格式的时间戳
    now = datetime.now()
    date = format_date_time(mktime(now.timetuple()))

    # 拼接字符串
    signature_origin = "host: " + host + "\n"
    signature_origin += "date: " + date + "\n"
    signature_origin += "POST " + path + " HTTP/1.1"

    # 进行hmac-sha256进行加密
    signature_sha = hmac.new(apiSecret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()

    signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

    authorization_origin = f'api_key="{apiKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

    # 将请求的鉴权参数组合为字典
    v = {
        "authorization": authorization,
        "date": date,
        "host": host
    }
    # 拼接鉴权参数，生成url
    reUrl = url + '?' + urlencode(v)
    # print(reUrl)
    # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
    return reUrl


def get_headers(url):
    headers = {
        'content-type': "application/json",
        'host': urlparse(url).netloc,
        'app_id': appid
    }
    return headers


def gen_create_request_data(text):
    data = {
        "header": {
            "app_id": appid,
            "status": 3,
            "channel": "default",
            "callback_url": "default",

        },
        "parameter": {
            "oig": {
                "result": {
                    "encoding": "utf8",
                    "compress": "raw",
                    "format": "json"
                },

            }
        },
        "payload": {
            "oig": {
                "text": text
            },
        },
    }
    return data


def create_task():
    text = {
        # "image": [image],  #引擎上传的原图，如果仅用图片生成能力，该字段需为空
        "prompt": "生成一个正在学习的二次元人物",  # 该prompt 可以是要求引擎生成的描述，也可以结合上传的图片要求模型修改原图
        "aspect_ratio": "1:1",
        "negative_prompt": "",
        "img_count": 4,
        "resolution": "2k"
    }
    b_text = base64.b64encode(json.dumps(text).encode("utf-8")).decode()
    request_url = create_url(create_host_url)
    data = gen_create_request_data(b_text)
    headers = get_headers(create_host_url)
    response = requests.post(request_url, data=json.dumps(data), headers=headers)
    # print(json.dumps(data))
    # return
    print('onMessage：\n' + response.text)
    resp = json.loads(response.text)
    taskid = resp['header']['task_id']
    # print(taskid)
    return taskid


def query_task(taskID):
    data = {
        "header": {
            "app_id": appid,
            "task_id": taskID  # 填写创建任务时返回的task_id
        }
    }
    request_url = create_url(query_host_url)
    headers = get_headers(query_host_url)
    response = requests.post(request_url, data=json.dumps(data), headers=headers)
    res = json.loads(response.content)

    return res


if __name__ == '__main__':
    # 创建任务
    task_id = create_task()

    # 查询结果 task_status 1：待处理 2：处理中 3：处理完成 4：回调完成
    while True:
        print(datetime.now())
        res = query_task(task_id)
        code = res["header"]["code"]
        task_status = ''
        if code == 0:
            task_status = res["header"]["task_status"]
            if ('' == task_status):
                print("查询任务状态有误，请检查")
            elif ('3' == task_status):
                print(datetime.now())
                print("任务完成")
                print(res)
                f_text = res["payload"]["result"]["text"]
                print("图片信息：\n" + str(base64.b64decode(f_text)))
                break
            else:
                print("查询任务中：......" + json.dumps(res))
                time.sleep(1)
                continue
        else:
            print(res)
