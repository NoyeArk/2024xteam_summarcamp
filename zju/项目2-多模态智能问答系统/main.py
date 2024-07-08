import re
import json
import hmac
import time
import base64
import hashlib
import requests
from PIL import Image
from io import BytesIO
from urllib import parse
from datetime import datetime
from matplotlib import pyplot as plt
from urllib.parse import urlencode, urlparse
from wsgiref.handlers import format_date_time


class MutilGPT:
    def __init__(self):
        # 模型1的API
        self.API_KEY = "1zvVmImc5oVwyAZDvWJkD2Fz"
        self.SECRET_KEY = "de5UGEFZ4CnhO4IiG1RRC4eEc7bS4ccP"
        # 模型2的API
        self.appid = "698c8614"
        self.apiSecret = "NDJhOTIzODljY2ZiMDk4OTg5MmZjYzI4"
        self.apiKey = "2fb3adb3314e207f14c4fce53e82b67a"
        # 请求地址
        # 请求地址
        self.create_host_url = "https://cn-huadong-1.xf-yun.com/v1/private/s3fd61810/create"
        self.query_host_url = "https://cn-huadong-1.xf-yun.com/v1/private/s3fd61810/query"

    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": self.API_KEY, "client_secret": self.SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))

    def model1(self, promote):
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/yi_34b_chat?access_token=" + self.get_access_token()

        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": promote
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        data = json.loads(response.text)
        return data['result']

    def build_auth_request_url(self, request_url):
        url_result = parse.urlparse(request_url)
        date = "Thu, 09 May 2024 02:09:13 GMT"  # format_date_time(mktime(datetime.now().timetuple()))
        print(date)
        method = "POST"
        signature_origin = "host: {}\ndate: {}\n{} {} HTTP/1.1".format(url_result.hostname, date, method, url_result.path)
        signature_sha = hmac.new(self.apiSecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.apiKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        values = {
            "host": url_result.hostname,
            "date": date,
            "authorization": authorization
        }
        return request_url + "?" + urlencode(values)

    def create_url(self, url):
        host = urlparse(url).netloc
        path = urlparse(url).path

        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(time.mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "POST " + path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.apiSecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.apiKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

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

    def get_headers(self, url):
        headers = {
            'content-type': "application/json",
            'host': urlparse(url).netloc,
            'app_id': self.appid
        }
        return headers

    def gen_create_request_data(self, text):
        data = {
            "header": {
                "app_id": self.appid,
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

    def query_task(self, taskID):
        data = {
            "header": {
                "app_id": self.appid,
                "task_id": taskID  # 填写创建任务时返回的task_id
            }
        }
        request_url = self.create_url(self.query_host_url)
        headers = self.get_headers(self.query_host_url)
        response = requests.post(request_url, data=json.dumps(data), headers=headers)
        res = json.loads(response.content)

        return res

    def create_task(self, request):
        text = {
            "prompt": request,
            "aspect_ratio": "1:1",
            "negative_prompt": "",
            "img_count": 1,
            "resolution": "2k"
        }
        b_text = base64.b64encode(json.dumps(text).encode("utf-8")).decode()
        request_url = self.create_url(self.create_host_url)
        data = self.gen_create_request_data(b_text)
        headers = self.get_headers(self.create_host_url)
        response = requests.post(request_url, data=json.dumps(data), headers=headers)
        print('onMessage：\n' + response.text)
        resp = json.loads(response.text)
        taskid = resp['header']['task_id']
        return taskid

    def model2(self, promote):
        # 创建任务
        task_id = self.create_task(promote)

        # 查询结果 task_status 1：待处理 2：处理中 3：处理完成 4：回调完成
        while True:
            print(datetime.now())
            res = self.query_task(task_id)
            code = res["header"]["code"]
            task_status = ''
            if code == 0:
                task_status = res["header"]["task_status"]
                if '' == task_status:
                    print("查询任务状态有误，请检查")
                elif '3' == task_status:
                    print(datetime.now())
                    print("任务完成")
                    print(res)
                    f_text = res["payload"]["result"]["text"]
                    f_text = str(base64.b64decode(f_text))
                    # print("图片信息：\n" + str(base64.b64decode(f_text)))
                    print("图片信息：\n" + f_text)
                    return f_text
                else:
                    print("查询任务中：......" + json.dumps(res))
                    time.sleep(1)
                    continue
            else:
                print(res)

    def get(self, promote):
        output = self.model1(promote)
        print(output)

        _str = self.model2(output)

        # 使用正则表达式匹配image_wm的值
        match = re.search(r'"image_wm":"([^"]*)"', _str)

        image_url = match.group(1)

        # 使用requests下载图片
        response = requests.get(image_url)

        # 确保请求成功
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            plt.figure('image')
            plt.imshow(img)
            plt.axis('off')  # 隐藏坐标轴
            plt.show()
        else:
            print("获取图片失败...")


if __name__ == '__main__':
    model = MutilGPT()
    # response = model.model1("延吉")
    # print(response)
    # model.get("延吉")
    model._test_model2()
