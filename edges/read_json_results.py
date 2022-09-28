import json


def snr_on_second(filename: str, second: int):
    with open(filename, 'r') as f:
        json_content = f.read()
    result_dict = json.loads(json_content)
    needed_snrs = []
    for i in result_dict:
        needed_snrs.append(round(result_dict[i][second], 3))
        if i == "20":
            break
    return needed_snrs


if __name__ == "__main__":
    print(snr_on_second('motion_signals_examples/signal_snr_statistics_3_3_1.json', 0))
