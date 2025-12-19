import os
import cv2
import pathlib
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class ChangeDetection:
    """
    - detect.py 에서 전달한 seated_count, waiting_count 를 사용해서
      전체 인원 / 좌석 수 비율로 혼잡도 상태(여유/보통/혼잡)를 계산.
    - 혼잡도 상태가 '바뀌었을 때만' 서버에 한 번 업로드.
    - 상태가 잠깐 튀는 것 막기 위해, 같은 상태가 N 프레임 이상 유지됐을 때만
      상태 변경으로 인정.
    """

    HOST = os.getenv("DJANGO_HOST")
    username = os.getenv("DJANGO_USERNAME")
    password = os.getenv("DJANGO_PASSWORD")

    # 좌석 수 (없으면 기본 80석)
    TOTAL_SEATS = int(os.getenv("TOTAL_SEATS", "80"))

    # 혼잡도 기준 (비율)
    LOW_THRESHOLD = float(os.getenv("LOW_THRESHOLD", "0.4"))   # < 0.4  -> 여유
    HIGH_THRESHOLD = float(os.getenv("HIGH_THRESHOLD", "0.8")) # < 0.8  -> 보통, 그 이상은 혼잡

    # 상태가 바뀌었다고 인정하기 위해 필요한 최소 프레임 수
    STATUS_STABLE_FRAMES = int(os.getenv("STATUS_STABLE_FRAMES", "10"))

    def __init__(self):
        self.token = None

        # 최근 프레임들의 상태 추적
        self.current_status = None          # 지금 연속으로 유지 중인 상태
        self.status_frame_count = 0         # 그 상태가 몇 프레임째 이어지는지

        # 마지막으로 "서버에 보냈던" 상태
        self.last_sent_status = None

        print("[ChangeDetection] HOST    :", self.HOST)
        print("[ChangeDetection] USERNAME:", self.username)
        print("[ChangeDetection] PASSWORD:", "(HIDDEN)" if self.password else None)
        print("[ChangeDetection] TOTAL_SEATS:", self.TOTAL_SEATS)

        if not self.HOST or not self.username or not self.password:
            raise RuntimeError(
                "[ChangeDetection] .env 설정(DJANGO_HOST / DJANGO_USERNAME / DJANGO_PASSWORD)을 확인하세요."
            )

        # 토큰 발급
        login_url = self.HOST.rstrip("/") + "/api-token-auth/"
        print("[Login URL]", login_url)

        res = requests.post(
            login_url,
            {
                "username": self.username,
                "password": self.password,
            },
        )
        print("[ChangeDetection] status:", res.status_code)
        print("[ChangeDetection] body  :", res.text)

        res.raise_for_status()
        self.token = res.json().get("token")
        print("[ChangeDetection] Token:", self.token)

    # ------------------------- 상태 계산 -------------------------

    def _compute_status(self, total_people: int) -> str:
        """전체 인원 / 좌석 수 비율로 상태 문자열 반환."""
        if self.TOTAL_SEATS <= 0:
            return "unknown"

        ratio = total_people / float(self.TOTAL_SEATS)

        if ratio < self.LOW_THRESHOLD:
            return "여유"
        elif ratio < self.HIGH_THRESHOLD:
            return "보통"
        else:
            return "혼잡"

    # ----------------------- 메인 업데이트 ------------------------

    def add(self, current_ids, save_dir, image, seated_count: int, waiting_count: int):
        """
        detect.py 에서 프레임마다 호출.

        current_ids     : PersonTracker가 준 현재 프레임의 ID 집합 (지금은 크게 사용 X)
        save_dir        : YOLO run 폴더 (이미지 저장 경로 계산용)
        image           : 현재 프레임 (numpy array, BGR)
        seated_count    : 앉은 사람 수
        waiting_count   : 서 있는 사람 수(대기열)
        """
        now = datetime.now()
        now_str = now.isoformat()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")

        total_people = int(seated_count) + int(waiting_count)

        if self.TOTAL_SEATS <= 0:
            return

        status = self._compute_status(total_people)

        # ---- 1) 프레임 단위 상태 안정화 ----
        if status == self.current_status:
            self.status_frame_count += 1
        else:
            # 상태가 바뀌었으니 새 상태로 초기화
            self.current_status = status
            self.status_frame_count = 1

        # 아직 충분히 오래 유지되지 않았으면 "변경"으로 보지 않음
        if self.status_frame_count < self.STATUS_STABLE_FRAMES:
            return

        # ---- 2) 서버에 마지막으로 보냈던 상태와 같으면 업로드 X ----
        if status == self.last_sent_status:
            return

        # 여기까지 왔으면:
        # - status 가 STATUS_STABLE_FRAMES 프레임 이상 유지됐고
        # - last_sent_status 와 다르다  => 진짜로 상태가 바뀌었다고 판단
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
            f"[STATUS CHANGE] status={status}, total_people={total_people}, "
            f"seated={seated_count}, waiting={waiting_count}"
        )

        self._send_one(save_dir, image, title, text, now_str)

    # ---------------------- 업로드 함수 ---------------------------

    def _send_one(self, save_dir, image, title, text, now_str):
        """
        한 번의 POST 업로드 (이미지 1장 + 제목/본문)
        """
        now = datetime.now()

        base = pathlib.Path(os.getcwd())
        save_path = (
            base / save_dir / "detected" /
            str(now.year) / str(now.month) / str(now.day)
        )
        save_path.mkdir(parents=True, exist_ok=True)

        filename = f"{now.hour}-{now.minute}-{now.second}-{now.microsecond}.jpg"
        full_path = save_path / filename

        # 이미지 리사이즈 후 저장 (너무 크지 않게)
        dst = cv2.resize(image, dsize=(320, 240), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(full_path), dst)

        headers = {
            "Authorization": "JWT " + self.token,
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
                data=data,
                files=files,
                headers=headers,
            )

        print("[Upload]", res.status_code)
        print("[Response]", res.text)
