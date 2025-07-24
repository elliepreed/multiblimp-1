import pandas as pd

def score_tse(model, fn: str):
    tse_df = pd.read_csv(fn, sep="\t")

    tse_df["sen_prob"] = pd.Series(dtype=object)
    tse_df["wrong_prob"] = pd.Series(dtype=object)

    max_length = 512  # default context window; safe for GPT-style models

    for idx, row in tse_df.iterrows():
        try:
            sen_prob, wrong_prob = score_pair(model, row.sen, row.wrong_sen, max_length)

            sen_nll = -sen_prob.sum().item()
            wrong_nll = -wrong_prob.sum().item()

            tse_df.at[idx, "sen_prob"] = sen_prob.tolist()
            tse_df.at[idx, "wrong_prob"] = wrong_prob.tolist()
            tse_df.loc[idx, "sen_nll"] = sen_nll
            tse_df.loc[idx, "wrong_nll"] = wrong_nll
            tse_df.loc[idx, "delta"] = wrong_nll - sen_nll

        except Exception as e:
            print(f"Error scoring row {idx}: {e}")
            tse_df.at[idx, "sen_prob"] = None
            tse_df.at[idx, "wrong_prob"] = None
            tse_df.loc[idx, "sen_nll"] = None
            tse_df.loc[idx, "wrong_nll"] = None
            tse_df.loc[idx, "delta"] = None

    return tse_df


def score_pair(ilm_model, sen, wrong_sen, max_length=512):
    stimuli = [sen, wrong_sen]

    return ilm_model.sequence_score(
        stimuli,
        reduction=lambda x: x,
        truncation=True,
        max_length=max_length
    )
