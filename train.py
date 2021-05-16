import argparse
import json
import os
import random
os.environ['WANDB_PROJECT'] = 'Pstage3_DST'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup
import wandb

from data_utils import (WOSDataset, get_examples_from_dialogues, load_dataset, set_seed,
                        seed_everything)
from eval_utils import DSTEvaluator
from evaluation import _evaluation
from inference import inference
from model_transformer import TRADE, masked_cross_entropy_for_value
from preprocessor import TRADEPreprocessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../input/data/train_dataset")
    parser.add_argument("--model_dir", type=str, default="TRADE_v3")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--warmup_ratio", type=int, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--word_dropout", type=int, default=0)
    ################ transformer decoder로 추가된 argument.
    parser.add_argument("--max_position", type=int, help='허용 가능한 input token의 최대 갯수' ,default=512) 
    parser.add_argument("--attention_drop_out", type=int, default=0.1)
    parser.add_argument("--num_attention_heads", type=int, help='hidden_size는 head갯수로 나뉠 수 있어야 한다.', default=6)
    parser.add_argument("--ffn_dim", type=int, default=768*2)
    parser.add_argument("--num_decoder_layers", type=int, default=6)
    ################ transformer decoder로 추가된 argument.
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Subword Vocab만을 위한 huggingface model",
        default='dsksd/bert-ko-small-minimal',
    )

    # Model Specific Argument
    parser.add_argument("--hidden_size", type=int, help="디코더의 hidden size", default=768)
    parser.add_argument(
        "--vocab_size",
        type=int,
        help="vocab size, subword vocab tokenizer에 의해 특정된다",
        default=None,
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--proj_dim", type=int,
                        help="만약 지정되면 기존의 hidden_size는 embedding dimension으로 취급되고, proj_dim이 GRU의 hidden_size로 사용됨. hidden_size보다 작아야 함.", default=None)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    args = parser.parse_args()

    wandb.init(tags=[f'BERT encoder={args.model_name_or_path}', 'add yes,no slot', f'word_dropout {args.word_dropout}', 'Transformer Decoder'], name = args.model_dir)

    
    # args.data_dir = os.environ['SM_CHANNEL_TRAIN']
    # args.model_dir = os.environ['SM_MODEL_DIR']

    # random seed 고정
    set_seed(args.random_seed)
    # seed_everything(args.random_seed)

    # Data Loading
    train_data_file = f"{args.data_dir}/train_dials.json"
    slot_meta = json.load(open(f"{args.data_dir}/slot_meta.json"))
    '''['관광-경치 좋은', '관광-교육적', '관광-도보 가능', '관광-문화 예술', '관광-역사적', '관광-이름', '관광-종류', '관광-주차 가능', '관광-지역', '숙소-가격대', '숙소-도보 가능', '숙소-수영장 유무', '숙소-스파 유무', '숙소-예약 기간', '숙소-예약 명수', '숙소-예약 요일', '숙소-이름', '숙소-인터넷 가능', '숙소-조식 가능', '숙소-종류', '숙소-주차 가능', '숙소-지역', '숙소-헬스장 유무', '숙소-흡연 가능', '식당-가격대', '식당-도보 가능', '식당-야외석 유무', '식당-예약 명수', '식당-예약 시간', '식당-예약 요일', '식당-이름', '식당-인터넷 가능', '식당-종류', '식당-주류 판매', '식당-주차 가능', '식당-지역', '식당-흡연 가능', '지하철-도착지', '지하철-출발 시간', '지하철-출발지', '택시-도착 시간', '택시-도착지', '택시-종류', '택시-출발 시간', '택시-출발지']'''
    train_data, dev_data, dev_labels = load_dataset(train_data_file)

    train_examples = get_examples_from_dialogues(
        train_data, user_first=False, dialogue_level=False
    )
    '''
    train_examples[10] = DSTInputExample(guid='polished-poetry-0057:관광_9-2', context_turns=['', '쇼핑을 하려는데 서울 서쪽에 있을까요?', '서울 서쪽에 쇼핑이 가능한 곳이라면 노량진 수산물 도매시장이 있습니다.', '오 네 거기 주소 좀 알려주세요.'], current_turn=['노량진 수산물 도매시장의 주소는 서울 동작구 93806입니다.', '알려주시는김에 연락처랑 평점도 좀 알려주세요.'], label=['관광-종류-쇼핑', '관광-지역-서울 서쪽', '관광-이름-노량진 수산물 도매시장'])
    '''
    dev_examples = get_examples_from_dialogues(
        dev_data, user_first=False, dialogue_level=False
    )
    '''
    dev_examples[10] = DSTInputExample(guid='shy-sea-4716:관광_11-2', context_turns=['', '제가 서울을 처음 와봐서 문화 예술과 관련된 곳으로 관광하고 싶은데 어디로 가면 될까요?', '안녕하세요. 관광을 원하시는 지역과 종류가 있으시면 말씀해주세요.', '서울 중앙쪽으로 알려주세요. 종류는 글쎄요 잘 모르겠어요.'], current_turn=['네. 그럼 명동난타극장과 삼성미술관 라움, 정동극장을 추천해드립니다. 어디가 괜찮으세요?', '미술관이 좋을것 같아요. 지하철로 이동하려고 하는데 어디에서 내리면 되나요?'], label=None)
    '''

    # Define Preprocessor
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    processor = TRADEPreprocessor(slot_meta, tokenizer, word_dropout_rate = args.word_dropout)
    args.vocab_size = len(tokenizer)
    args.n_gate = len(processor.gating2id) # gating 갯수 none, dontcare, ptr

    # Extracting Featrues
    train_features = processor.convert_examples_to_features(train_examples)
    '''train_features[2]
    OpenVocabDSTFeature(guid='snowy-hat-8324:관광_식당_11-2', input_id=[2, 3, 6265, 6672, 4073, 3249, 4034, 8732, 4292, 6722, 4076, 8553, 3, 11655, 4279, 8553, 18, 6336, 4481, 22014, 6771, 4204, 4112, 8538, 4147, 27233, 35, 18790, 4086, 24, 4469, 10749, 14043, 4006, 4073, 4325, 3311, 4112, 6392, 4110, 2734, 4219, 3249, 4576, 6216, 18, 3, 3311, 4116, 4150, 7149, 18790, 4112, 2633, 4151, 4076, 5240, 4050, 6698, 4467, 4029, 4070, 13177, 4479, 4065, 4150, 35, 3, 6698, 4467, 4029, 4034, 9908, 26885, 11684, 25845, 4204, 10561, 18, 2373, 6289, 4279, 4147, 2054, 3249, 4154, 4161, 10397, 35, 3, 2279, 13090, 4192, 2024, 4112, 6249, 4234, 15532, 4403, 4292, 2010, 4219, 4451, 4112, 4244, 4150, 11431, 4221, 4007, 3249, 16868, 4479, 4150, 3], segment_id=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], gating_id=[0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0], target_ids=[[21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [6336, 4481, 22014, 6771, 4204, 3], [8732, 3, 0, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [6265, 6672, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [93, 6756, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [15532, 4403, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [6265, 6672, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0]])
    '''
    dev_features = processor.convert_examples_to_features(dev_examples)
    '''dev_features[2]
    OpenVocabDSTFeature(guid='wild-bonus-5601:식당_택시_12-2', input_id=[2, 3, 11655, 4279, 8553, 18, 6265, 10097, 4073, 8117, 4070, 6259, 4283, 26713, 4403, 4292, 3430, 4219, 3249, 4576, 6216, 18, 3, 11655, 4279, 8553, 18, 8863, 6243, 29365, 4034, 27672, 4034, 13177, 2411, 4114, 4065, 4150, 35, 3, 27672, 4034, 14053, 18781, 4150, 18, 3234, 18, 11139, 4147, 10472, 4110, 6477, 4279, 4034, 2084, 10749, 6465, 8161, 10756, 18, 3, 2279, 18, 3084, 5012, 4576, 6216, 18, 8863, 6265, 16417, 4050, 4073, 6767, 4283, 15119, 4083, 4007, 30524, 2084, 4112, 8538, 4147, 27233, 35, 18790, 4112, 24, 18, 23, 10749, 6465, 4608, 6216, 18, 3, 11946, 4279, 17164, 6479, 3757, 4467, 4172, 6304, 7090, 4151, 4076, 4114, 5012, 7933, 35, 3], segment_id=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], gating_id=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], target_ids=[[21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3]])

--> gating_id가 전부 0이고 taret_id가 전부 똑같다(전부 'none'임).

dev_labels['wild-bonus-5601:식당_택시_12-2'] = ['식당-가격대-dontcare', '식당-지역-서울 북쪽', '식당-종류-중식당', '식당-주차 가능-yes', '식당-주류 판매-yes']

    '''
    # Slot Meta tokenizing for the decoder initial inputs
    tokenized_slot_meta = []
    for slot in slot_meta:
        tokenized_slot_meta.append(
            tokenizer.encode(slot.replace("-", " "), add_special_tokens=False)
        )
    '''각 domain-slot pair를 토크나이징 한다.
    tokenized_slot_meta[0] = [6728, 21170, 3311, 4112]
    tokenized_slot_meta[1] = [6728, 6295, 4199, 0]
    tokenized_slot_meta[2] = [6728, 17502, 6259, 0]
    '''
    
    # Model 선언
    model = TRADE(args, tokenized_slot_meta)
    # model.set_subword_embedding(args.model_name_or_path)  # Subword Embedding 초기화
    # print(f"Subword Embeddings is loaded from {args.model_name_or_path}")
    model.to(device)
    print("Model is initialized")

    train_data = WOSDataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        sampler=train_sampler,
        collate_fn=processor.collate_fn,
    )
    '''collate_fn에서 한 배치안에서 가장 긴 input_id의 길이를 기준으로 다른 input_id를 패딩한다. 
    '''
    print("# train:", len(train_data))

    dev_data = WOSDataset(dev_features)
    dev_sampler = SequentialSampler(dev_data)
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.eval_batch_size,
        sampler=dev_sampler,
        collate_fn=processor.collate_fn,
    )
    print("# dev:", len(dev_data))
    
    # Optimizer 및 Scheduler 선언
    n_epochs = args.num_train_epochs
    t_total = len(train_loader) * n_epochs
    warmup_steps = int(t_total * args.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    loss_fnc_1 = masked_cross_entropy_for_value  # generation
    loss_fnc_2 = nn.CrossEntropyLoss()  # gating

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    json.dump(
        vars(args),
        open(f"{args.model_dir}/exp_config.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    json.dump(
        slot_meta,
        open(f"{args.model_dir}/slot_meta.json", "w"),
        indent=2,
        ensure_ascii=False,
    )
    
    best_score, best_checkpoint = 0, 0
    for epoch in range(n_epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids, segment_ids, input_masks, gating_ids, target_ids, guids = [
                b.to(device) if not isinstance(b, list) else b for b in batch
            ] ## b가 list가 아니면 b.to(device)를 하고 b가 list면 그냥 b를 쓴다.
            '''
            ipdb>  input_ids
            tensor([[    2,     3, 11655,  ...,     0,     0,     0],
                    [    2,     3, 13756,  ...,     0,     0,     0],
                    [    2,     3, 11655,  ...,     0,     0,     0],
                    [    2,     3, 11655,  ...,  4150,    18,     3]], device='cuda:0')
            
            
            ipdb>  target_ids[0]
            tensor([[21832, 11764,     3,     0,     0],
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
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [33922,  4019, 21172,  7139,     3],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [20762,  4403,     3,     0,     0],
            [   93,  6756,     3,     0,     0],
            [   93,  6756,     3,     0,     0],
            [ 6265, 10806,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0],
            [21832, 11764,     3,     0,     0]], device='cuda:0')
            
            ipdb>  input_ids.size()
            torch.Size([4, 258])
            ipdb>  segment_ids.size()
            torch.Size([4, 258])
            ipdb>  input_masks.size()
            torch.Size([4, 258])
            ipdb>  gating_ids.size()
            torch.Size([4, 45]) 
            ipdb>  target_ids.size()
            torch.Size([4, 45, 5])
            '''
            
#             # teacher forcing
#             if (
#                 args.teacher_forcing_ratio > 0.0
#                 and random.random() < args.teacher_forcing_ratio
#             ):
#                 tf = target_ids
#             else:
#                 tf = None

#             all_point_outputs, all_gate_outputs = model(
#                 input_ids, segment_ids, input_masks, target_ids.size(-1), tf
#             )
            all_point_outputs, all_gate_outputs = model(
                input_ids, target_ids, segment_ids, input_masks, 
            )
            
            # generation loss
            loss_1 = loss_fnc_1(
                all_point_outputs.contiguous(),
                target_ids.contiguous().view(-1),
                tokenizer.pad_token_id,
            )
            
            # gating loss
            loss_2 = loss_fnc_2(
                all_gate_outputs.contiguous().view(-1, args.n_gate),
                gating_ids.contiguous().view(-1),
            )
            loss = loss_1 + loss_2

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                print(
                    f"[{epoch}/{n_epochs}] [{step}/{len(train_loader)}] loss: {loss.item()} gen: {loss_1.item()} gate: {loss_2.item()}"
                )
                wandb.log(
                {
                    "total loss": loss.item(),
                    "value generation loss": loss_1.item(),
                    "slot gate loss": loss_2.item(),

                })
            

        predictions = inference(model, dev_loader, processor, device)
        eval_result = _evaluation(predictions, dev_labels, slot_meta)
        for k, v in eval_result.items():
            print(f"{k}: {v}")
            wandb.log(
            {
                f"{k}": v,
            })

        if best_score < eval_result['joint_goal_accuracy']:
            print("Update Best checkpoint!")
            best_score = eval_result['joint_goal_accuracy']
            best_checkpoint = epoch

        torch.save(model.state_dict(), f"{args.model_dir}/model-{epoch}.bin")
    print(f"Best checkpoint: {args.model_dir}/model-{best_checkpoint}.bin")
