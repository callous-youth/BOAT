import abc


class DynamicalSystem(object):
    def __init__(self, ll_objective, ul_objective, lower_loop, ul_model, ll_model, solver_config) -> None:
        """
        Implements the abstract class for the lower-level optimization procedure.

        Parameters
        ----------
            :param ll_objective: The lower-level objective of the BLO problem.
            :type ll_objective: callable
            :param ul_objective: The upper-level objective of the BLO problem.
            :type ul_objective: callable
            :param ll_model: The lower-level model of the BLO problem.
            :type ll_model: torch.nn.Module
            :param ul_model: The upper-level model of the BLO problem.
            :type ul_model: torch.nn.Module
            :param lower_loop: Number of iterations for lower-level optimization.
            :type lower_loop: int
        """
        self.ll_objective = ll_objective
        self.ul_objective = ul_objective
        self.lower_loop = lower_loop
        self.ul_model = ul_model
        self.ll_model = ll_model
        self.solver_config = solver_config

    @abc.abstractmethod
    def optimize(self, **kwargs):
        pass
