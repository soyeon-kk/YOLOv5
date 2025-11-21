import os
import pathlib
from datetime import datetime

import cv2
import requests
from dotenv import load_dotenv

load_dotenv()


class ChangeDetection:
    result_prev = []

    HOST = os.getenv("DJANGO_HOST")
    username = os.getenv("DJANGO_USERNAME")
    password = os.getenv("DJANGO_PASSWORD")
    token = ""
    title = ""
    text = ""

    def __init__(self, names):
        self.result_prev = [0 for _ in range(len(names))]

        # .env에서 읽힌 값 확인용 디버그 출력
        print("[ChangeDetection] HOST    :", self.HOST)
        print("[ChangeDetection] USERNAME:", self.username)
        print("[ChangeDetection] PASSWORD:", "(숨김)" if self.password else None)

        if not self.HOST or not self.username or not self.password:
            print(
                "[ChangeDetection] ❌ .env 설정이 비어 있음. DJANGO_HOST / DJANGO_USERNAME / DJANGO_PASSWORD 확인 필요"
            )
            # 더 진행하면 어차피 에러 나니까 여기서 정지
            raise RuntimeError("환경변수(.env)가 제대로 설정되지 않았습니다.")

        res = requests.post(
            self.HOST + "/api-token-auth/",
            {
                "username": self.username,
                "password": self.password,
            },
        )

        print("[ChangeDetection] status:", res.status_code)
        print("[ChangeDetection] body  :", res.text)

        # 여기서 바로 에러 던지기 전에, 응답 내용을 한 번 찍어보고,
        # 그래도 200이 아니면 raise
        res.raise_for_status()

        self.token = res.json()["token"]
        print("[ChangeDetection] Token:", self.token)

    def add(self, names, detected_current, save_dir, image):
        self.title = ""
        self.text = ""
        change_flag = 0

        for i in range(len(self.result_prev)):
            if self.result_prev[i] == 0 and detected_current[i] == 1:
                change_flag = 1
                self.title = names[i]
                self.text += names[i] + ", "

        self.result_prev = detected_current[:]

        if change_flag == 1:
            self.send(save_dir, image)

    def send(self, save_dir, image):
        now = datetime.now()
        now_str = now.isoformat()

        today = datetime.now()
        base = pathlib.Path(os.getcwd())

        save_path = base / save_dir / "detected" / str(today.year) / str(today.month) / str(today.day)

        save_path.mkdir(parents=True, exist_ok=True)

        full_path = save_path / f"{today.hour}-{today.minute}-{today.second}-{today.microsecond}.jpg"

        dst = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(full_path), dst)

        headers = {"Authorization": "JWT " + self.token, "Accept": "application/json"}

        data = {"title": self.title, "text": self.text, "created_date": now_str, "published_date": now_str}

        with open(full_path, "rb") as f:
            files = {"image": f}
            res = requests.post(self.HOST + "/api_root/Post/", data=data, files=files, headers=headers)

        print("Upload status:", res.status_code)
        print("Response:", res.text)
