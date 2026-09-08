FLAG_THRESHOLD_CER = 0.10

def flag_for_review(cer: float) -> bool:
    return cer >= FLAG_THRESHOLD_CER