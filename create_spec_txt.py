

#%%
with open("spec_file.txt", mode="w+") as file:
    file.write("enc_key: 'nvidia_tlt'")


# %%
import os

CURRENT_DIR = os.getcwd()
# %%
with open(".env", "w+") as env_file:
    env_file.write(f"CURRENT_DIR={CURRENT_DIR}")

# %%
