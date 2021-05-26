# TRADE

## decoder를 GRU에서 Transformer의 decoder로 변경
 - 기존 TRADE에서는 value를 generation할 때 이전의 예측한 토큰을 잊거나 dialouge context를 잊는 듯한 모습이 보였음.
 - p^history를 통해 context에 대한 정보를 명시적으로 최종 확률분포에 반영하긴 했으나 이걸로는 부족해 보임.
 - Transformer의 decoder는 target id끼리의 self attention과 encoder-decoder-attention이 있기 때문에 위의 문제를 보완할 수 있을 것으로 기대.

