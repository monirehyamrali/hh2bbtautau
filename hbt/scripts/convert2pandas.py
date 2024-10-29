import glob

import pandas as pd
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


PATTERN = "/nfs/dust/cms/user/yamralim/cf_cache/hbt_store/analysis_hbt/cf.UniteColumns/run2_2017_nano_uhh_v11/{}/nominal/calib__none/sel__default/prod__z_fractions/dev3_allz/data_*.parquet"

dfs = []

for process_name, df_name in [
    ("dy_lep_pt100To250_amcatnlo", "dy"),
    ("tt_dl_powheg", "tt"),
    ("hh_ggf_bbtautau_madgraph", "hh"),
]:
    for path in glob.glob(PATTERN.format(process_name)):
        data = ak.from_parquet(path)
        #from IPython import embed; embed()
        data = ak.from_parquet(path, columns=["z_gen_pos", "z_gen_neg", "z_rec_neg", "z_rec_pos", "dm_neg", "dm_pos", "Tau.mass", "tau_nus", "tautauNN_regression_output"])
        df = ak.to_dataframe(data)
        df["process"] = df_name
        dfs.append(df)

df = pd.concat(dfs)


fig, ax = plt.subplots()


sns.jointplot(
    data=df,
    # data=df[(df["dm_pos"] == 1) & (df["dm_neg"] == 1)],
    x="z_rec_pos",
    y="z_rec_neg",
    hue="process",
    # kind="kde",
    ax=ax,
)

fig.savefig("test.pdf")