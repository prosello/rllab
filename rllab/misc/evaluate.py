import numpy as np
from rllab.misc import tensor_utils
from rllab.misc import special
from rllab.sampler.ma_sampler import cent_rollout, dec_rollout, conc_rollout


def evaluate(env, agent, max_path_length, n_paths, ma_mode, disc):
    if ma_mode == 'centralized':
        ret = []
        discret = []
        envinfo = []
        for n_path in range(n_paths):
            path = cent_rollout(env, agent, max_path_length)
            pathret = path['rewards'].sum()
            pathdiscret = special.discount_cumsum(path['rewards'], disc)
            info = path['env_infos']
            ret.append(pathret)
            discret.append(pathdiscret)
            envinfo.append(info)

        dictinfo = {k: np.mean(v) for k, v in tensor_utils.stack_tensor_dict_list(envinfo).items()}
        return dict(ret=np.mean(ret), discret=np.mean(discret), **dictinfo)

    elif ma_mode == 'decentralized':
        agent2paths = {}
        for agid in range(len(env.agents)):
            agent2paths[agid] = []

        for n_path in range(n_paths):
            paths = dec_rollout(env, agent, max_path_length)
            for agid, agpath in enumerate(paths):
                agent2paths[agid].append(agpath)

        rets, retsstd, discrets, infos = [], [], [], []
        for agid, paths in agent2paths.items():
            rets.append(np.mean([path['rewards'].sum() for path in paths]))
            retsstd.append(np.std([path['rewards'].sum() for path in paths]))
            discrets.append(
                np.mean([special.discount_cumsum(path['rewards'], disc)[0] for path in paths]))
            infos.append({
                k: np.mean(v)
                for k, v in tensor_utils.stack_tensor_dict_list(
                    [path['env_infos'] for path in paths]).items()
            })
            dictinfos = tensor_utils.stack_tensor_dict_list(infos)
        return dict(ret=rets, retstd=retsstd, discret=discrets, **dictinfos)
    elif ma_mode == 'concurrent':
        raise NotImplementedError()
