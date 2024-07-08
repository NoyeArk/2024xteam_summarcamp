import time

import requests
import json

API_KEY = "1zvVmImc5oVwyAZDvWJkD2Fz"
SECRET_KEY = "de5UGEFZ4CnhO4IiG1RRC4eEc7bS4ccP"


def main():
    access_token = get_access_token()
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/yi_34b_chat?access_token=" + access_token

    # 假设我们要发送多个请求
    messages = ["你好", "今天天气怎么样？", "谢谢"]

    for message in messages:
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.post(url, headers=headers, data=payload)
        response_text = response.text
        print("Received:", response_text)  # 立即输出响应

        # 尝试解析JSON（注意：如果响应不是有效的JSON，这将抛出异常）
        try:
            data = json.loads(response_text)
            print("Data:", data.get('result', "No 'result' field in response"))
        except json.JSONDecodeError:
            print("Failed to decode JSON response")

            # 可以在这里添加延时来模拟实时数据接收的间隔
        time.sleep(1)


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':
    main()
