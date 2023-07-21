import pandas as pd
# import modin.pandas as pd
import numpy as np
from scipy.ndimage.interpolation import shift
from tqdm import tqdm

MAX_ACTION = 150
data_folder = "/data1/zhxue/kuairand/data/KuaiRand-27K/data/"

# ================== read data ======================
print("loading data...")
try:
    selected_video_ids = pd.read_csv(data_folder+"selected_video_ids_27k.csv", index_col=[0])
    selected_log = pd.read_csv(data_folder+"selected_log_27k.csv", index_col=[0])
except:
    print("No cached selected logs")
    df_long = pd.read_csv(data_folder+"log_standard_4_08_to_4_21_27k_part1.csv")
    selected_video_ids = None
    selected_log = None
print("loaded logs.")

df_user_features = pd.read_csv(data_folder+"user_features_27k.csv")
print("loaded user features.")

try:
    selected_video_selected_features = pd.read_csv(data_folder+"selected_video_selected_features_27k.csv", index_col=[0])
except:
    print("No cached selected videos")
    # No cached data. Read original video features, which can be slow!
    df_video_features = pd.read_csv(data_folder+"video_features_basic_27k.csv")
    df_video_features_static = pd.read_csv(data_folder+"video_features_statistic_27k_part1.csv")
    selected_video_selected_features = None
print("loaded video features.")

# =================== select logs with extensively viewed videos ============
if selected_video_ids is None:
    grouped = df_long.groupby('video_id').count()
    selected_indexes = np.argsort(np.array(grouped['user_id']))[::-1][:150]
    selected_video_ids = grouped['user_id'].index[selected_indexes]
    selected_log = df_long[df_long['video_id'].isin(selected_video_ids)]

# ==================== select states with preliminary features ===========
# state for user information
print("Preparing states")
state_user = df_user_features.iloc[selected_log['user_id']]\
            [['user_active_degree', 
              'is_live_streamer', 
              'is_video_author', 
              'follow_user_num_range',
              'fans_user_num_range',
              'register_days_range',
              *['onehot_feat%d'%i for i in range(18)]]]

# turn str feature into int
for feature_name in ['user_active_degree', 
              'is_live_streamer', 
              'is_video_author', 
              'follow_user_num_range',
              'fans_user_num_range',
              'register_days_range']:
    trans_dict = dict(zip(state_user[feature_name].value_counts().index, range(len(state_user[feature_name].value_counts()))))
    state_user[feature_name] = state_user[feature_name].apply(lambda x: trans_dict[x])

# state for static video information. static means 150 static videos
if selected_video_selected_features is None:
    # no cached video features
    selected_video_features_static = df_video_features_static.iloc[selected_video_ids]
    selected_video_features = df_video_features.iloc[selected_video_ids]
    selected_video_selected_features = selected_video_features[["video_duration", 
                                                                "music_type", 
                                                                "tag"]]
    selected_video_selected_features['tag'] = selected_video_selected_features['tag'].apply(lambda x: int(x.split(",")[0]))
    selected_video_selected_features['music_type'].fillna(2, inplace=True)

    selected_video_selected_features_statistic = selected_video_features_static[['play_user_num', 'play_duration', 'like_cnt']]
    selected_video_selected_features = selected_video_selected_features.join(selected_video_selected_features_statistic)

state_static_video = np.stack([np.array(selected_video_selected_features).reshape(-1)] * 480385)

# state for dynamic video information which changes as the user views.
video_pool_size = 20
state_dynamic_video = np.empty((480385, video_pool_size, 6))
viewed_video_id = [None] * video_pool_size
viewed_video_pointer = -1
cur_user_id = 0
index = -1
print("Generating dynamic video information...")
try:
    with open(data_folder+"state_dynamic_video.npy", 'rb') as f:
        state_dynamic_video = np.load(f)
except:
    print("No cached state for dynamic video")
    for _, row in tqdm(selected_log.iterrows()):  
        index += 1
        viewed_video_pointer += 1
        user_id = int(row['user_id'])
        video_id = int(row['video_id'])
        if user_id != cur_user_id:
            # new user
            viewed_video_id = [None] * video_pool_size
            viewed_video_pointer = 0
            cur_user_id = user_id
            
        if viewed_video_pointer >= video_pool_size:
            viewed_video_pointer = 0

        viewed_video_id[viewed_video_pointer] = video_id
        current_state_dynamic_video = [[0]*6 if video_id is None else selected_video_selected_features.loc[video_id] for video_id in viewed_video_id]
        state_dynamic_video[index] = np.array(current_state_dynamic_video)

    with open(data_folder+"state_dynamic_video.npy", "wb+") as f:
        np.save(f, state_dynamic_video)

state_dynamic_video = state_dynamic_video.reshape(480385, -1)
# state_dynamic_video.to_csv("home/xuezhenghai/ks-constrained-rl-rs/data/kuairand_data/KuaiRand-27K/data/state_dynamic_video.csv")
state = np.hstack([state_user, state_static_video, state_dynamic_video])

next_state = np.vstack([state[1:], state[-1]])

# dones are True when the user changes
# Preparing dones and actions
user_id = np.array(selected_log['user_id'])
user_id_shift = shift(user_id, -1, cval=-1)
done = user_id != user_id_shift

# actions are one-dimentional intergers in range [0, MAX_ACTION]
selected_video_id_list = list(selected_video_ids.values.reshape(-1))
actions = np.array(selected_log['video_id'].apply(lambda x: selected_video_id_list.index(x)))
# select rewards with preliminary standards
rewards = np.array(selected_log[['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'is_hate', 'play_time_ms']])
rewards[:, -1] = rewards[:, -1]/10000
print("Parsing data ok. Saving to disk...")
train_num = 450000
small_num = 10000
with open(data_folder+'parsed_data/train_states.npy', 'wb+') as f:
    np.save(f, state[:train_num])
with open(data_folder+'parsed_data/train_next_states.npy', 'wb+') as f:
    np.save(f, next_state[:train_num])
with open(data_folder+'parsed_data/train_actions.npy', 'wb+') as f:
    np.save(f, actions[:train_num])
with open(data_folder+'parsed_data/train_rewards.npy', 'wb+') as f:
    np.save(f, rewards[:train_num])
with open(data_folder+'parsed_data/train_dones.npy', 'wb+') as f:
    np.save(f, done[:train_num])

with open(data_folder+'parsed_data/small_train_states.npy', 'wb+') as f:
    np.save(f, state[:small_num])
with open(data_folder+'parsed_data/small_train_next_states.npy', 'wb+') as f:
    np.save(f, next_state[:small_num])
with open(data_folder+'parsed_data/small_train_actions.npy', 'wb+') as f:
    np.save(f, actions[:small_num])
with open(data_folder+'parsed_data/small_train_rewards.npy', 'wb+') as f:
    np.save(f, rewards[:small_num])
with open(data_folder+'parsed_data/small_train_dones.npy', 'wb+') as f:
    np.save(f, done[:small_num])

with open(data_folder+'parsed_data/test_states.npy', 'wb+') as f:
    np.save(f, state[train_num:])
with open(data_folder+'parsed_data/test_next_states.npy', 'wb+') as f:
    np.save(f, next_state[train_num:])
with open(data_folder+'parsed_data/test_actions.npy', 'wb+') as f:
    np.save(f, actions[train_num:])
with open(data_folder+'parsed_data/test_rewards.npy', 'wb+') as f:
    np.save(f, rewards[train_num:])
with open(data_folder+'parsed_data/test_dones.npy', 'wb+') as f:
    np.save(f, done[train_num:])
