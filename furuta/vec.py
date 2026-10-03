"""1 プロセスに複数の環境を載せる VecEnv（プロセス間通信の回数を減らして学習を速くする）."""

import multiprocessing as mp

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def _worker(remote, env_fns):
    envs = [f() for f in env_fns]
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, rews, dones, infos = [], [], [], []
            for env, a in zip(envs, data):
                o, r, term, trunc, info = env.step(a)
                done = term or trunc
                info["TimeLimit.truncated"] = trunc and not term
                if done:
                    info["terminal_observation"] = o
                    o, reset_info = env.reset()
                obs.append(o)
                rews.append(r)
                dones.append(done)
                infos.append(info)
            remote.send((np.stack(obs), np.array(rews, np.float32), np.array(dones), infos))
        elif cmd == "reset":
            remote.send(np.stack([env.reset(seed=s)[0] for env, s in zip(envs, data)]))
        elif cmd == "spaces":
            remote.send((envs[0].observation_space, envs[0].action_space))
        elif cmd == "get_attr":
            remote.send([getattr(env, data) for env in envs])
        elif cmd == "close":
            remote.close()
            break


class BatchedSubprocVecEnv(VecEnv):
    def __init__(self, env_fns, n_workers):
        self.chunks = [list(c) for c in np.array_split(np.arange(len(env_fns)), n_workers)]
        ctx = mp.get_context("forkserver")
        self.remotes, self.procs = [], []
        for c in self.chunks:
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, [env_fns[i] for i in c]), daemon=True)
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.remotes[0].send(("spaces", None))
        obs_space, act_space = self.remotes[0].recv()
        super().__init__(len(env_fns), obs_space, act_space)

    def reset(self):
        seeds = self._seeds if any(s is not None for s in self._seeds) else [None] * self.num_envs
        for r, c in zip(self.remotes, self.chunks):
            r.send(("reset", [seeds[i] for i in c]))
        obs = np.concatenate([r.recv() for r in self.remotes])
        self._reset_seeds()
        return obs

    def step_async(self, actions):
        for r, c in zip(self.remotes, self.chunks):
            r.send(("step", actions[c]))

    def step_wait(self):
        res = [r.recv() for r in self.remotes]
        obs = np.concatenate([x[0] for x in res])
        rews = np.concatenate([x[1] for x in res])
        dones = np.concatenate([x[2] for x in res])
        infos = [i for x in res for i in x[3]]
        return obs, rews, dones, infos

    def close(self):
        for r in self.remotes:
            r.send(("close", None))
        for p in self.procs:
            p.join()

    def get_attr(self, attr_name, indices=None):
        out = []
        for r in self.remotes:
            r.send(("get_attr", attr_name))
            out += r.recv()
        return out if indices is None else [out[i] for i in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
