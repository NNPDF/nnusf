# -*- coding: utf-8 -*-
import json

from hyperopt import Trials, space_eval


def space_eval_trial(space, trial):
    """Define the trial within the space."""
    for_eval = {}
    for k, v in trial["misc"]["vals"].items():
        if len(v) == 0:
            for_eval[k] = None
        else:
            for_eval[k] = v[0]
    return space_eval(space, for_eval)


class FileTrials(Trials):
    """Database interface supporting data-driven model-based optimization
    based on the Trials class of hyperopt.
    """

    def __init__(
        self,
        folder_path,
        name,
        log=None,
        parameters=None,
        exp_key=None,
        refresh=True,
    ):
        self._store_trial = False
        self._json_file = f"{folder_path}/{name}.json"
        self._parameters = parameters
        self.log_info = log.info if log else print
        super(FileTrials, self).__init__(exp_key=exp_key, refresh=refresh)

    def refresh(self):
        """Refresh the state of the trials."""
        super(FileTrials, self).refresh()

        if self._store_trial:
            local_trials = []
            for idx, t in enumerate(self._dynamic_trials):
                local_trials.append(t)
                tr = space_eval_trial(self._parameters, t)
                local_trials[idx]["misc"]["space_vals"] = tr

            josn_str = json.dumps(local_trials, default=str)
            with open(self._json_file, "w") as f:
                f.write(josn_str)

    def new_trial_ids(self, n):
        """Trial identification object within the Trial instances."""
        self._store_trial = False
        return super(FileTrials, self).new_trial_ids(n)

    def new_trial_docs(self, tids, specs, results, miscs):
        """Database documentation of the trials."""
        self._store_trial = True
        return super(FileTrials, self).new_trial_docs(
            tids, specs, results, miscs
        )
