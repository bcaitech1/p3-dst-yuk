import json
import argparse
from eval_utils import DSTEvaluator


SLOT_META_PATH = 'data/train_dataset/slot_meta.json'


def _evaluation(preds, labels, slot_meta):
    '''
    ipdb>  dev_labels['long-morning-7263:식당_관광_8-0']
    ['식당-가격대-저렴', '식당-지역-서울 중앙']
    ipdb>  predictions
    {'wild-bonus-5601:식당_택시_12-0': ['식당-종류-중식당', '식당-주차 가능-yes', '식당-지역-서울 북쪽'], 
    'wild-bonus-5601:식당_택시_12-1': ['식당-가격대-dontcare', '식당-종류-중식당', '식당-주류 판매-yes', 
    '식당-주차 가능-yes', '식당-지역-서울 북쪽']}
    '''
    evaluator = DSTEvaluator(slot_meta)

    evaluator.init()
    assert len(preds) == len(labels)

    for k, l in labels.items():
        p = preds.get(k)
        if p is None:
            raise Exception(f"{k} is not in the predictions!")
        evaluator.update(l, p)

    result = evaluator.compute()
    print(result)
    return result


def evaluation(gt_path, pred_path):
    slot_meta = json.load(open(SLOT_META_PATH))
    gts = json.load(open(gt_path))
    preds = json.load(open(pred_path))
    eval_result = _evaluation(preds, gts, slot_meta)
    return eval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--pred_path', type=str, required=True)
    args = parser.parse_args()
    eval_result = evaluation(args.gt_path, args.pred_path)
