import os
import cv2
import pathlib
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class ChangeDetection:
    """
    - seated_count, waiting_count로 혼잡도 계산
    - 상태가 N 프레임 이상 유지될 때만 변경으로 인정
    - 마지막으로 서버에 보낸 상태와 다를 때만 업로드
    """

    # ================= 환경 변수 =================
    HOST = os.getenv("DJANGO_HOST")              # 예: https://soyeonkk.pythonanywhere.com
    USERNAME = os.getenv("DJANGO_USERNAME")      # arsenic
    PASSWORD = os.getenv("DJANGO_PASSWORD")      # 비밀번호

    TOTAL_SEATS = int(os.getenv("TOTAL_SEATS", "80"))

    LOW_THRESHOLD = float(os.getenv("LOW_THRESHOLD", "0.4"))
    HIGH_THRESHOLD = float(os.getenv("HIGH_THRESHOLD", "0.8"))

    STATUS_STABLE_FRAMES = int(os.getenv("STATUS_STABLE_FRAMES", "10"))
    # ============================================

    def __init__(self):
        self.token = None

        self.current_status = None
        self.status_frame_count = 0
        self.last_sent_status = None

        print("[ChangeDetection] HOST        :", self.HOST)
        print("[ChangeDetection] USERNAME    :", self.USERNAME)
        print("[ChangeDetection] PASSWORD    :", "(HIDDEN)" if self.PASSWORD else None)
        print("[ChangeDetection] TOTAL_SEATS :", self.TOTAL_SEATS)

        if not all([self.HOST, self.USERNAME, self.PASSWORD]):
            raise RuntimeError("❌ .env 설정(DJANGO_HOST / USERNAME / PASSWORD) 확인 필요")

        # ---------- 토큰 발급 ----------
        login_url = self.HOST.rstrip("/") + "/api-token-auth/"
        print("[Login URL]", login_url)

        res = requests.post(
            login_url,
            data={
                "username": self.USERNAME,
                "password": self.PASSWORD,
            },
            timeout=10,
        )

        print("[Login Status]", res.status_code)
        print("[Login Body  ]", res.text)

        res.raise_for_status()

        self.token = res.json().get("token")
        if not self.token:
            raise RuntimeError("❌ 토큰 발급 실패")

        print("[ChangeDetection] Token OK")

    # ================= 상태 계산 =================

    def _compute_status(self, total_people: int) -> str:
        if self.TOTAL_SEATS <= 0:
            return "unknown"

        ratio = total_people / float(self.TOTAL_SEATS)

        if ratio < self.LOW_THRESHOLD:
            return "여유"
        elif ratio < self.HIGH_THRESHOLD:
            return "보통"
        else:
            return "혼잡"

    # ================= 메인 로직 =================

    def add(self, current_ids, save_dir, image, seated_count: int, waiting_count: int):
        now = datetime.now()
        now_str = now.isoformat()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        total_people = int(seated_count) + int(waiting_count)
        status = self._compute_status(total_people)

        # ---- 상태 안정화 ----
        if status == self.current_status:
            self.status_frame_count += 1
        else:
            self.current_status = status
            self.status_frame_count = 1

        if self.status_frame_count < self.STATUS_STABLE_FRAMES:
            return

        # ---- 중복 업로드 방지 ----
        if status == self.last_sent_status:
            return

        self.last_sent_status = status

        seats_left = max(self.TOTAL_SEATS - seated_count, 0)

        title = f"실시간 혼잡도 변경 - {status}"
        text = (
            f"{time_str} 기준 학생식당 혼잡도 상태가 '{status}'로 변경되었습니다.\n"
            f"- 총 좌석 수: {self.TOTAL_SEATS}석\n"
            f"- 착석 인원: {seated_count}명\n"
            f"- 대기열 인원(서 있는 인원): {waiting_count}명\n"
            f"- 남은 좌석: {seats_left}석"
        )

        print(
            f"[STATUS CHANGE] {status} | seated={seated_count}, waiting={waiting_count}"
        )

        self._send_one(save_dir, image, title, text, now_str)

    # ================= 서버 업로드 =================

    def _send_one(self, save_dir, image, title, text, now_str):
        now = datetime.now()

        base = pathlib.Path(os.getcwd())
        save_path = base / save_dir / "detected" / str(now.year) / str(now.month) / str(now.day)
        save_path.mkdir(parents=True, exist_ok=True)

        filename = f"{now.hour}-{now.minute}-{now.second}-{now.microsecond}.jpg"
        full_path = save_path / filename

        resized = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(full_path), resized)

        headers = {
            "Authorization": f"Token {self.token}",   # 🔥 중요
            "Accept": "application/json",
        }

        data = {
            "title": title,
            "text": text,
            "created_date": now_str,
            "published_date": now_str,
        }

        with open(full_path, "rb") as f:
            files = {"image": f}
            res = requests.post(
                self.HOST.rstrip("/") + "/api_root/Post/",
                headers=headers,
                data=data,
                files=files,
                timeout=15,
            )

        print("[Upload Status]", res.status_code)
        print("[Upload Body  ]", res.text)
