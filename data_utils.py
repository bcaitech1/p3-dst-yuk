import dataclasses
import json
import random
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class OntologyDSTFeature: ## SUMBT같은 ontology based 모델을 위한 feature 클래스
    guid: str
    input_ids: List[int]
    segment_ids: List[int]
    num_turn: int
    target_ids: Optional[List[int]]


@dataclass
class OpenVocabDSTFeature: ## TRADE같은 open vocab based 모델을 위한 feature 클래스
    guid: str
    input_id: List[int]
    segment_id: List[int]
    gating_id: List[int]
    target_ids: Optional[Union[List[int], List[List[int]]]]


class WOSDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx]


def load_dataset(dataset_path, dev_split=0.1):
    data = json.load(open(dataset_path))
    '''
    ipdb>  data[0]
{'dialogue_idx': 'snowy-hat-8324:관광_식당_11', 'domains': ['관광', '식당'], 'dialogue': [{'role': 'user', 'text': '서울 중앙에 있는 박물관을 찾아주세요', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙']}, {'role': 'sys', 'text': '안녕하세요. 문화역서울 284은 어떠신가요? 평점도 4점으로 방문객들에게 좋은 평가를 받고 있습니다.'}, {'role': 'user', 'text': '좋네요 거기 평점은 말해주셨구 전화번호가 어떻게되나요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284']}, {'role': 'sys', 'text': '전화번호는 983880764입니다. 더 필요하신 게 있으실까요?'}, {'role': 'user', 'text': '네 관광지와 같은 지역의 한식당을 가고싶은데요 야외석이 있어야되요', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes']}, {'role': 'sys', 'text': '생각하고 계신 가격대가 있으신가요?'}, {'role': 'user', 'text': '음.. 저렴한 가격대에 있나요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-저렴', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes']}, {'role': 'sys', 'text': '죄송하지만 저렴한 가격대에는 없으시네요.'}, {'role': 'user', 'text': '그럼 비싼 가격대로 다시 찾아주세요', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes']}, {'role': 'sys', 'text': '외계인의맛집은 어떠신가요? 대표 메뉴는 한정식입니다.'}, {'role': 'user', 'text': '좋습니당 토요일 18:00에 1명 예약가능한가요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes', '식당-예약 요일-토요일', '식당-예약 시간-18:00', '식당-예약 명수-1', '식당-이름-외계인의맛집']}, {'role': 'sys', 'text': '가능합니다. 예약도와드릴까요?'}, {'role': 'user', 'text': '넹 거기 주류는 판매하나요?주차는 가능한가요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes', '식당-예약 요일-토요일', '식당-예약 시간-18:00', '식당-예약 명수-1', '식당-이름-외계인의맛집']}, {'role': 'sys', 'text': '주류는 판매하고 있고 주차도 가능합니다. 더 궁금하신 점 있으신가요?'}, {'role': 'user', 'text': '아니용', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes', '식당-예약 요일-토요일', '식당-예약 시간-18:00', '식당-예약 명수-1', '식당-이름-외계인의맛집']}, {'role': 'sys', 'text': '감사합니다. 즐거운 여행되세요.'}]}
    '''
    num_data = len(data) ## 7000
    num_dev = int(num_data * dev_split) ## 700
    if not num_dev:
        return data, []  # no dev dataset

    dom_mapper = defaultdict(list)
    for d in data:
        dom_mapper[len(d["domains"])].append(d["dialogue_idx"])
    '''
    ipdb>  dom_mapper
defaultdict(<class 'list'>, {2: ['snowy-hat-8324:관광_식당_11', 'calm-smoke-9954:관광_식당_7', 'wild-bonus-5601:식당_택시_12'], 1: ['polished-poetry-0057:관광_9', 'falling-king-2544:택시_1'], 3: ['autumn-mountain-9993:식당_관광_지하철_14']})
    '''
    num_per_domain_trainsition = int(num_dev / 3) ## 233
    '''len(dom_mapper) = 3이다. len(dom_mapper)는 한 dialogue내의 도메인 개수를 의미함.
    num_per_domain_trainsition은 int(num_dev / 3)인데 이는 도메인 개수가 1개인 dialogue, domain개수가 2개인 dialouge, 도메인 개수가 3개인 dialouge를 각각 num_dev/3 개씩 뽑아서 dev_set으로 쓰려는 것임.
    '''
    dev_idx = []
    for v in dom_mapper.values():
        idx = random.sample(v, num_per_domain_trainsition) ## v로부터 num_per_domain_trainsition개 만큼을 샘플링한다.
        '''ipdb>  idx
['divine-voice-7296:관광_식당_12', 'tiny-night-7495:식당_숙소_7', 'nameless-sky-7549:숙소_관광_8', 'red-morning-6153:숙소_식당_12', 'icy-frost-1896:식당_택시_7', 'late-bar-4292:숙소_식당_14', 'shy-bread-7497:숙소_택시_9', 'tiny-salad-4509:식당_관광_9', 'red-union-8784:숙소_식 ... ]'''
        dev_idx.extend(idx)

    train_data, dev_data = [], []
    for d in data:
        if d["dialogue_idx"] in dev_idx:
            dev_data.append(d)
        else:
            train_data.append(d)

    dev_labels = {}
    for dialogue in dev_data:
        d_idx = 0
        guid = dialogue["dialogue_idx"] ## 'long-morning-7263:식당_관광_8'
        for idx, turn in enumerate(dialogue["dialogue"]):
            if turn["role"] != "user": ## system 발화부분에서는 state가 기록되어있지 않으므로 넘어감.
                continue

            state = turn.pop("state") ## pop하면 해당 key와 value는 딕셔너리에서 없어짐. 그래서 dav_data의 각 dialouge에는 state가 없다.
            '''['식당-가격대-저렴', '식당-지역-서울 중앙']'''

            guid_t = f"{guid}-{d_idx}" 
            '''long-morning-7263:식당_관광_8-0 
            d_idx는 해당 dialogue안에서 해당 trun이 몇번째 turn인지를 의미함.
            '''
            d_idx += 1

            dev_labels[guid_t] = state
    '''
    ipdb>  dev_data[0]
{'dialogue_idx': 'long-morning-7263:식당_관광_8', 'domains': ['식당', '관광'], 'dialogue': [{'role': 'user', 'text': '친구랑 여행 계획을 세우는 중인데 서울 중앙에 저렴한 식당이 있으면 예약하고 싶어요.'}, {'role': 'sys', 'text': '안녕하세요. 원하시는 음식 종류가 어떻게 되세요?'}, {'role': 'user', 'text': '아무거나 괜찮습니다.', 'state': ['식당-가격대-저렴', '식당-지역-서울 중앙', '식당-종류-dontcare']}, {'role': 'sys', 'text': '그럼 서대문구에 있는 오동통규동동이라는 일식당은 어떠실까요? 가격 만원대로 저렴한 곳입니다.'}, {'role': 'user', 'text': '아 괜찮네요. 그럼 거기 토요일 6시에 가려고 하는데 2명 예약되나요?', 'state': ['식당-가격대-저렴', '식당-지역-서울 중앙', '식당-종류-dontcare', '식당-예약 요일-토요일', '식당-예약 시간-06:00', '식당-예약 명수-2', '식당-이름-오동통규동동']}, {'role': 'sys', 'text': '네. 가능하여 예약진행해드렸습니다. 예약 번호는 CSDG0입니다.'}, {'role': 'user', 'text': '감사합니다. 식당 영업 시간도 좀 알려주세요.', 'state': ['식당-가격대-저렴', '식당-지역-서울 중앙', '식당-종류-dontcare', '식당-예약 요일-토요일', '식당-예약 시간-06:00', '식당-예약 명수-2', '식당-이름-오동통규동동']}, {'role': 'sys', 'text': '영업 시간은 10:00~21:00입니다. 더 궁금한건 없으세요?'}, {'role': 'user', 'text': '서울 남쪽에 동물원이 있다는 이야길 들어서 가보려고 하는데 어딘지 알 수 있을까요?', 'state': ['관광-종류-동물원', '관광-지역-서울 남쪽', '식당-가격대-저렴', '식당-지역-서울 중앙', '식당-종류-dontcare', '식당-예약 요일-토요일', '식당-예약 시간-06:00', '식당-예약 명수-2', '식당-이름-오동통규동동']}, {'role': 'sys', 'text': '강남구에 코엑스 아쿠아리움이 있는데 이 곳을 말씀하시는게 맞을까요?'}, {'role': 'user', 'text': '맞는것 같아요. 거기 경치 좋은 곳인가요?', 'state': ['관광-종류-동물원', '관광-지역-서울 남쪽', '관광-이름-코엑스 아쿠아리움', '식당-가격대-저렴', '식당-지역-서울 중앙', '식당-종류-dontcare', '식당-예약 요일-토요일', '식당-예약 시간-06:00', '식당-예약 명수-2', '식당-이름-오동통규동동']}, {'role': 'sys', 'text': '실내라서 경치있는 곳은 아닙니다. 더 필요한건 없으신가요?'}, {'role': 'user', 'text': '없습니다. 알려주셔서 감사합니다.', 'state': ['관광-종류-동물원', '관광-지역-서울 남쪽', '관광-이름-코엑스 아쿠아리움', '식당-가격대-저렴', '식당-지역-서울 중앙', '식당-종류-dontcare', '식당-예약 요일-토요일', '식당-예약 시간-06:00', '식당-예약 명수-2', '식당-이름-오동통규동동']}, {'role': 'sys', 'text': '네. 그럼 이용해 주셔서 감사합니다.'}]}
### 첫번째 turn의 state만 pop하고 확인한거라 다른 turn의 state는 아직 있는상태이다. ###
ipdb>  dev_labels['long-morning-7263:식당_관광_8-0']
['식당-가격대-저렴', '식당-지역-서울 중앙']
ipdb>  train_data[0]
{'dialogue_idx': 'snowy-hat-8324:관광_식당_11', 'domains': ['관광', '식당'], 'dialogue': [{'role': 'user', 'text': '서울 중앙에 있는 박물관을 찾아주세요', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙']}, {'role': 'sys', 'text': '안녕하세요. 문화역서울 284은 어떠신가요? 평점도 4점으로 방문객들에게 좋은 평가를 받고 있습니다.'}, {'role': 'user', 'text': '좋네요 거기 평점은 말해주셨구 전화번호가 어떻게되나요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284']}, {'role': 'sys', 'text': '전화번호는 983880764입니다. 더 필요하신 게 있으실까요?'}, {'role': 'user', 'text': '네 관광지와 같은 지역의 한식당을 가고싶은데요 야외석이 있어야되요', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes']}, {'role': 'sys', 'text': '생각하고 계신 가격대가 있으신가요?'}, {'role': 'user', 'text': '음.. 저렴한 가격대에 있나요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-저렴', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes']}, {'role': 'sys', 'text': '죄송하지만 저렴한 가격대에는 없으시네요.'}, {'role': 'user', 'text': '그럼 비싼 가격대로 다시 찾아주세요', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes']}, {'role': 'sys', 'text': '외계인의맛집은 어떠신가요? 대표 메뉴는 한정식입니다.'}, {'role': 'user', 'text': '좋습니당 토요일 18:00에 1명 예약가능한가요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes', '식당-예약 요일-토요일', '식당-예약 시간-18:00', '식당-예약 명수-1', '식당-이름-외계인의맛집']}, {'role': 'sys', 'text': '가능합니다. 예약도와드릴까요?'}, {'role': 'user', 'text': '넹 거기 주류는 판매하나요?주차는 가능한가요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes', '식당-예약 요일-토요일', '식당-예약 시간-18:00', '식당-예약 명수-1', '식당-이름-외계인의맛집']}, {'role': 'sys', 'text': '주류는 판매하고 있고 주차도 가능합니다. 더 궁금하신 점 있으신가요?'}, {'role': 'user', 'text': '아니용', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes', '식당-예약 요일-토요일', '식당-예약 시간-18:00', '식당-예약 명수-1', '식당-이름-외계인의맛집']}, {'role': 'sys', 'text': '감사합니다. 즐거운 여행되세요.'}]}

    '''
    return train_data, dev_data, dev_labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_slot(dom_slot_value, get_domain_slot=False):
    try:
        dom, slot, value = dom_slot_value.split("-")
    except ValueError:
        tempo = dom_slot_value.split("-")
        if len(tempo) < 2:
            return dom_slot_value, dom_slot_value, dom_slot_value
        dom, slot = tempo[0], tempo[1]
        value = dom_slot_value.replace(f"{dom}-{slot}-", "").strip()

    if get_domain_slot:
        return f"{dom}-{slot}", value
    return dom, slot, value


def build_slot_meta(data):
    slot_meta = []
    for dialog in data:
        for turn in dialog["dialogue"]:
            if not turn.get("state"):
                continue

            for dom_slot_value in turn["state"]:
                domain_slot, _ = split_slot(dom_slot_value, get_domain_slot=True)
                if domain_slot not in slot_meta:
                    slot_meta.append(domain_slot)
    return sorted(slot_meta)


def convert_state_dict(state):
    dic = {}
    for slot in state:
        s, v = split_slot(slot, get_domain_slot=True)
        dic[s] = v
    return dic


@dataclass
class DSTInputExample:
    guid: str
    context_turns: List[str]
    current_turn: List[str]
    label: Optional[List[str]] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def get_examples_from_dialogue(dialogue, user_first=False):
    '''load_dataset으로 load한 데이터들 각각을 DSTInputExample로 바꿔준다.
    ipdb>  dialogue
{'dialogue_idx': 'snowy-hat-8324:관광_식당_11', 'domains': ['관광', '식당'], 'dialogue': [{'role': 'user', 'text': '서울 중앙에 있는 박물관을 찾아주세요', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙']}, {'role': 'sys', 'text': '안녕하세요. 문화역서울 284은 어떠신가요? 평점도 4점으로 방문객들에게 좋은 평가를 받고 있습니다.'}, {'role': 'user', 'text': '좋네요 거기 평점은 말해주셨구 전화번호가 어떻게되나요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284']}, {'role': 'sys', 'text': '전화번호는 983880764입니다. 더 필요하신 게 있으실까요?'}, {'role': 'user', 'text': '네 관광지와 같은 지역의 한식당을 가고싶은데요 야외석이 있어야되요', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes']}, {'role': 'sys', 'text': '생각하고 계신 가격대가 있으신가요?'}, {'role': 'user', 'text': '음.. 저렴한 가격대에 있나요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-저렴', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes']}, {'role': 'sys', 'text': '죄송하지만 저렴한 가격대에는 없으시네요.'}, {'role': 'user', 'text': '그럼 비싼 가격대로 다시 찾아주세요', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes']}, {'role': 'sys', 'text': '외계인의맛집은 어떠신가요? 대표 메뉴는 한정식입니다.'}, {'role': 'user', 'text': '좋습니당 토요일 18:00에 1명 예약가능한가요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes', '식당-예약 요일-토요일', '식당-예약 시간-18:00', '식당-예약 명수-1', '식당-이름-외계인의맛집']}, {'role': 'sys', 'text': '가능합니다. 예약도와드릴까요?'}, {'role': 'user', 'text': '넹 거기 주류는 판매하나요?주차는 가능한가요?', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes', '식당-예약 요일-토요일', '식당-예약 시간-18:00', '식당-예약 명수-1', '식당-이름-외계인의맛집']}, {'role': 'sys', 'text': '주류는 판매하고 있고 주차도 가능합니다. 더 궁금하신 점 있으신가요?'}, {'role': 'user', 'text': '아니용', 'state': ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes', '식당-예약 요일-토요일', '식당-예약 시간-18:00', '식당-예약 명수-1', '식당-이름-외계인의맛집']}, {'role': 'sys', 'text': '감사합니다. 즐거운 여행되세요.'}]}
    '''
    guid = dialogue["dialogue_idx"]
    '''snowy-hat-8324:관광_식당_11'''
    examples = []
    history = [] ## dialogue context. 모든 유저,시스템발화를 순서대로 저장한다.
    d_idx = 0
    for idx, turn in enumerate(dialogue["dialogue"]):
        if turn["role"] != "user":
            continue

        if idx:
            sys_utter = dialogue["dialogue"][idx - 1]["text"] ## 바로 전(idx - 1)의 system 발화를 가져온다.
        else: ## idx=0 인 첫번째 turn에는 이전 system 발화는 없으므로 sys_utter = ""로 한다.
            sys_utter = ""

        user_utter = turn["text"]
        '''서울 중앙에 있는 박물관을 찾아주세요'''
        state = turn.get("state")
        '''['관광-종류-박물관', '관광-지역-서울 중앙']'''
        context = deepcopy(history)
        if user_first:
            current_turn = [user_utter, sys_utter]
        else:
            current_turn = [sys_utter, user_utter]
        examples.append(
            DSTInputExample(
                guid=f"{guid}-{d_idx}",
                context_turns=context,
                current_turn=current_turn,
                label=state,
            )
        )
        history.append(sys_utter)
        history.append(user_utter)
        d_idx += 1
    '''ipdb>  examples[-1]
DSTInputExample(guid='snowy-hat-8324:관광_식당_11-7', context_turns=['', '서울 중앙에 있는 박물관을 찾아주세요', '안녕하세요. 문화역서울 284은 어떠신가요? 평점도 4점으로 방문객들에게 좋은 평가를 받고 있습니다.', '좋네요 거기 평점은 말해주셨구 전화번호가 어떻게되나요?', '전화번호는 983880764입니다. 더 필요하신 게 있으실까요?', '네 관광지와 같은 지역의 한식당을 가고싶은데요 야외석이 있어야되요', '생각하고 계신 가격대가 있으신가요?', '음.. 저렴한 가격대에 있나요?', '죄송하지만 저렴한 가격대에는 없으시네요.', '그럼 비싼 가격대로 다시 찾아주세요', '외계인의맛집은 어떠신가요? 대표 메뉴는 한정식입니다.', '좋습니당 토요일 18:00에 1명 예약가능한가요?', '가능합니다. 예약도와드릴까요?', '넹 거기 주류는 판매하나요?주차는 가능한가요?'], current_turn=['주류는 판매하고 있고 주차도 가능합니다. 더 궁금하신 점 있으신가요?', '아니용'], label=['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-비싼', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes', '식당-예약 요일-토요일', '식당-예약 시간-18:00', '식당-예약 명수-1', '식당-이름-외계인의맛집'])
    '''
    return examples


def get_examples_from_dialogues(data, user_first=False, dialogue_level=False):
    examples = []
    for d in tqdm(data):
        example = get_examples_from_dialogue(d, user_first=user_first)
        if dialogue_level:
            examples.append(example)
        else:
            examples.extend(example)
    return examples


class DSTPreprocessor:
    def __init__(self, slot_meta, src_tokenizer, trg_tokenizer=None, ontology=None):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology

    def pad_ids(self, arrays, pad_idx, max_length=-1):
        '''가장 길이가 긴 List와 같은 길이로 나머지 List에 패딩토큰을 추가.
        Args:
            arrays (List[List]) : 2 Dimensional List. each List have different length.
            pad_idx
            max_length
        '''
        if max_length < 0:
            max_length = max(list(map(len, arrays)))

        arrays = [array + [pad_idx] * (max_length - len(array)) for array in arrays]
        return arrays

    def pad_id_of_matrix(self, arrays, padding, max_length=-1, left=False):
        if max_length < 0:
            max_length = max([array.size(-1) for array in arrays])
        '''
        ipdb>  arrays
[tensor([[21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [10238,     3,     0,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [   25,     3,     0,     0,     0],
        [   22,     3,     0,     0,     0],
        [ 9826,     3,     0,     0,     0],
        [ 8139,  7396,  6265,     3,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [33922,  4019, 21172,  7139,     3],
        [21832, 11764,     3,     0,     0],
        [ 6265, 10806,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [24534, 14704,  4357,     3,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [ 8139,  7396,  6265,     3,     0],
        [33922,  4019, 21172,  7139,     3],
        [33922,  4019, 21172,  7139,     3],
        [24534, 14704,  4357,     3,     0]]), tensor([[21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [   93,  6756,     3],
        [21832, 11764,     3],
        [20227,     3,     0],
        [21832, 11764,     3],
        [ 6265,  6672,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3],
        [21832, 11764,     3]]), tensor([[21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0],
        [21832, 11764,     3,     0,     0], ...
        '''
        new_arrays = []
        for i, array in enumerate(arrays):
            n, l = array.size()
            pad = torch.zeros(n, (max_length - l))
            pad[
                :,
                :,
            ] = padding
            pad = pad.long()
            m = torch.cat([array, pad], -1)
            new_arrays.append(m.unsqueeze(0))

        return torch.cat(new_arrays, 0)

    def _convert_example_to_feature(self):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def recover_state(self):
        raise NotImplementedError
