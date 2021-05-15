import torch


from data_utils import DSTPreprocessor, OpenVocabDSTFeature, convert_state_dict


class TRADEPreprocessor(DSTPreprocessor):
    def __init__(
        self,
        slot_meta,
        src_tokenizer,
        trg_tokenizer=None,
        ontology=None,
        max_seq_length=512,
        word_dropout_rate = 0.1
    ):
        self.slot_meta = slot_meta
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer if trg_tokenizer else src_tokenizer
        self.ontology = ontology
        self.gating2id = {"none": 0, "dontcare": 1, "yes": 2, "no": 3, "ptr": 4}
        self.id2gating = {v: k for k, v in self.gating2id.items()}
        self.max_seq_length = max_seq_length
        self.word_dropout_rate = word_dropout_rate

    def _convert_example_to_feature(self, example):
        dialogue_context = " [SEP] ".join(example.context_turns + example.current_turn)
        '''
        ' [SEP] 서울 중앙에 있는 박물관을 찾아주세요'
        
        ' [SEP] 서울 중앙에 있는 박물관을 찾아주세요 [SEP] 안녕하세요. 문화역서울 284은 어떠신가요? 평점도 4점으로 방문객들에게 좋은 평가를 받고 있습니다. [SEP] 좋네요 거기 평점은 말해주셨구 전화번호가 어떻게되나요? [SEP] 전화번호는 983880764입니다. 더 필요하신 게 있으실까요? [SEP] 네 관광지와 같은 지역의 한식당을 가고싶은데요 야외석이 있어야되요 [SEP] 생각하고 계신 가격대가 있으신가요? [SEP] 음.. 저렴한 가격대에 있나요?'
        
        '''
        input_id = self.src_tokenizer.encode(dialogue_context, add_special_tokens=False)
        '''
        [3, 6265, 6672, 4073, 3249, 4034, 8732, 4292, 6722, 4076, 8553]
        
        [3, 6265, 6672, 4073, 3249, 4034, 8732, 4292, 6722, 4076, 8553, 3, 11655, 4279, 8553, 18, 6336, 4481, 22014, 6771, 4204, 4112, 8538, 4147, 27233, 35, 18790, 4086, 24, 4469, 10749, 14043, 4006, 4073, 4325, 3311, 4112, 6392, 4110, 2734, 4219, 3249, 4576, 6216, 18, 3, 3311, 4116, 4150, 7149, 18790, 4112, 2633, 4151, 4076, 5240, 4050, 6698, 4467, 4029, 4070, 13177, 4479, 4065, 4150, 35, 3, 6698, 4467, 4029, 4034, 9908, 26885, 11684, 25845, 4204, 10561, 18, 2373, 6289, 4279, 4147, 2054, 3249, 4154, 4161, 10397, 35, 3, 2279, 13090, 4192, 2024, 4112, 6249, 4234, 15532, 4403, 4292, 2010, 4219, 4451, 4112, 4244, 4150, 11431, 4221, 4007, 3249, 16868, 4479, 4150, 3, 6243, 4279, 4219, 12154, 27672, 4070, 3249, 4154, 4147, 27233, 35, 3, 3234, 18, 18, 8784, 4283, 27672, 4073, 3249, 4065, 4150, 35]
        '''
        max_length = self.max_seq_length - 2
        if len(input_id) > max_length:
            gap = len(input_id) - max_length
            input_id = input_id[gap:] ## max_length내 에서 가능한 한 최신 dialogue들을 가져온다.
        if self.word_dropout_rate:
            input_id = self.word_dropout(input_id)
        
        input_id = (
            [self.src_tokenizer.cls_token_id]
            + input_id
            + [self.src_tokenizer.sep_token_id]
        )
        '''
        [2, 3, 6265, 6672, 4073, 3249, 4034, 8732, 4292, 6722, 4076, 8553, 3]
        
        [2, 3, 6265, 6672, 4073, 3249, 4034, 8732, 4292, 6722, 4076, 8553, 3, 11655, 4279, 8553, 18, 6336, 4481, 22014, 6771, 4204, 4112, 8538, 4147, 27233, 35, 18790, 4086, 24, 4469, 10749, 14043, 4006, 4073, 4325, 3311, 4112, 6392, 4110, 2734, 4219, 3249, 4576, 6216, 18, 3, 3311, 4116, 4150, 7149, 18790, 4112, 2633, 4151, 4076, 5240, 4050, 6698, 4467, 4029, 4070, 13177, 4479, 4065, 4150, 35, 3, 6698, 4467, 4029, 4034, 9908, 26885, 11684, 25845, 4204, 10561, 18, 2373, 6289, 4279, 4147, 2054, 3249, 4154, 4161, 10397, 35, 3, 2279, 13090, 4192, 2024, 4112, 6249, 4234, 15532, 4403, 4292, 2010, 4219, 4451, 4112, 4244, 4150, 11431, 4221, 4007, 3249, 16868, 4479, 4150, 3, 6243, 4279, 4219, 12154, 27672, 4070, 3249, 4154, 4147, 27233, 35, 3, 3234, 18, 18, 8784, 4283, 27672, 4073, 3249, 4065, 4150, 35, 3]
        '''
        segment_id = [0] * len(input_id) ## 딱히 쓰이진 않음.그냥 None으로 둬도 될 것 같은데?

        target_ids = []
        gating_id = []
        if not example.label:
            example.label = []
        '''
        example.label = ['관광-종류-박물관', '관광-지역-서울 중앙']
        ['관광-종류-박물관', '관광-지역-서울 중앙', '관광-이름-문화역서울 284', '식당-가격대-저렴', '식당-지역-서울 중앙', '식당-종류-한식당', '식당-야외석 유무-yes']
        '''
        state = convert_state_dict(example.label)
        '''
        {'관광-종류': '박물관', '관광-지역': '서울 중앙'}
        {'관광-종류': '박물관', '관광-지역': '서울 중앙', '관광-이름': '문화역서울 284', '식당-가격대': '저렴', '식당-지역': '서울 중앙', '식당-종류': '한식당', '식당-야외석 유무': 'yes'}
        '''
        for slot in self.slot_meta:
            value = state.get(slot, "none") ## label에 없는 slot은 아직 언급되지 않았거나 회수된 것이므로 none으로 둔다.
            target_id = self.trg_tokenizer.encode(value, add_special_tokens=False) + [
                self.trg_tokenizer.sep_token_id
            ] ## value를 이루는 토큰드르이 index와 SEP토큰의 index오 이루어진 리스트. ex)[21832, 11764, 3]
            target_ids.append(target_id)
            gating_id.append(self.gating2id.get(value, self.gating2id["ptr"]))
            ## none이나 dont care가 아니라면 디코더의 출력값을 계속 트랙킹하는 것이므로 ptr의 index(=2)를 가져온다.
            
        '''
        target_ids = [[21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [6336, 4481, 22014, 6771, 4204, 3], [8732, 3], [21832, 11764, 3], [6265, 6672, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [8784, 3], [21832, 11764, 3], [93, 6756, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [15532, 4403, 3], [21832, 11764, 3], [21832, 11764, 3], [6265, 6672, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3], [21832, 11764, 3]]
        -> 각 slot에 대한 value의 토큰리스트들 이다.
        
        gating_id = [0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        -> 각 slot에 대한 gating label이다.
        
        -> turn 하나당 45개(slot 개수)의 target_id와 크기45인 gating_id(slot 하나당 gating_id 1개)가 있다.
        '''
        target_ids = self.pad_ids(target_ids, self.trg_tokenizer.pad_token_id) ## target_id에는 <sos>나 <eos>같은 토큰은 없음.
        '''가장 긴 토큰리스트를 가진 target_id에 맞춰서 패딩한다.
        [[21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [6336, 4481, 22014, 6771, 4204, 3], [8732, 3, 0, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [6265, 6672, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0], [21832, 11764, 3, 0, 0, 0]]
        '''
        return OpenVocabDSTFeature(
            example.guid, input_id, segment_id, gating_id, target_ids
        )

    def convert_examples_to_features(self, examples):
        return list(map(self._convert_example_to_feature, examples))

    def recover_state(self, gate_list, gen_list):
        assert len(gate_list) == len(self.slot_meta)
        assert len(gen_list) == len(self.slot_meta)

        recovered = []
        for slot, gate, value in zip(self.slot_meta, gate_list, gen_list):
            if self.id2gating[gate] == "none":
                continue

            if self.id2gating[gate] in ["dontcare", "yes", "no"]:
                recovered.append("%s-%s" % (slot, self.id2gating[gate]))
                continue

            token_id_list = []
            for id_ in value:
                if id_ in self.trg_tokenizer.all_special_ids:
                    break

                token_id_list.append(id_)
            value = self.trg_tokenizer.decode(token_id_list, skip_special_tokens=True)

            if value == "none":
                continue

            recovered.append("%s-%s" % (slot, value))
        return recovered
    
    def word_dropout(self, input_id):
        temp = torch.LongTensor(input_id)
        probs = torch.empty(len(temp)).uniform_(0, 1)
        temp = torch.where(probs > self.word_dropout_rate, temp, torch.empty(len(temp),dtype=torch.int64).fill_(self.src_tokenizer.unk_token_id))
        input_id = temp.tolist()

        return input_id
    
    def collate_fn(self, batch):
        '''
        Args:
            batch (List) :OpenVocabDSTFeature가 batch_size개수 만큼 들어있는 List.
        '''
        guids = [b.guid for b in batch]
        '''['jolly-block-3905:식당_30-4', 'lively-sound-8432:숙소_식당_2-3', 'small-rice-1651:식당_30-0', 'floral-voice-7626:식당_숙소_관광_21-1']'''
        input_ids = torch.LongTensor(
            self.pad_ids([b.input_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        segment_ids = torch.LongTensor(
            self.pad_ids([b.segment_id for b in batch], self.src_tokenizer.pad_token_id)
        )
        ## attention_mask를 collate_fn에서 생성한다.
        input_masks = input_ids.ne(self.src_tokenizer.pad_token_id) ## input_ids의 각 요소가 패딩토큰id와 다르면 True, 같으면 False -> [True, True, True, True,True, False, False, False]

        gating_ids = torch.LongTensor([b.gating_id for b in batch])
        target_ids = self.pad_id_of_matrix(
            [torch.LongTensor(b.target_ids) for b in batch],
            self.trg_tokenizer.pad_token_id,
        )
        '''target_id들 중 가장 긴것에 맞춰서 나머지들을 패딩하고 LongTensor로 만든다.
        torch.Size([4, 45, 5])'''
        return input_ids, segment_ids, input_masks, gating_ids, target_ids, guids
