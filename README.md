# Project Title

"How are you Today?"의 챗봇 서버입니다.

[huggingface에 파인튜닝된 요약모델](https://huggingface.co/HowAreYouToday/KoT5-summarization)

## Getting Started

프로젝트 사용법 입니다.

### Project Download

```bash
git clone https://github.com/HowAreUToday/tool.git
```

### Prerequisites

해당 프로젝트는 파이썬 3.8.18 버젼에서 구축됐습니다.
3.8.18이 아닌 3.8로 명시하여 다운로드시 langchain에서 패키지 관련 에러가 발생할 수 있습니다.

```bash
conda create --name MyProject python==3.8.18
```

### Installation and Usage

1. 관련 패키지 다운로드

```bash
pip install -r requirements.txt
```

2. env 파일 설정

openAI에서 발급받은 인증키를 입력하여 주세요.

3. 실행

```bash
python server.py
```

## License

이 프로젝트의 License관련 자세한 내용은 [LICENSE.md](LICENSE.md) 파일을 참조하십시오.

## References

이 프로젝트는 중 일부 T5 요약 모델은 paust_pkot팀의 사전학습 모델을 기반으로 학습됐습니다.

- [paust_pkot5_v1](https://github.com/paust-team/pko-t5) by Dennis Park (2022)

## Author & Acknowledgments

프로젝트에 기여한 정하연, 홍순빈, 유찬영님에게 대한 감사의 인사를 표시합니다.

- [정하연](https://hayeon.hashnode.dev/)
- [홍순빈](mailto:sb.hong0317@gmail.com)
- [유찬영](http://youngchannel.co.kr/)