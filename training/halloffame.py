class SnapShot:
    """
    Snapshot of a model and its performance.
    """
    def __init__(self, G, D, G_loss, D_loss, epoch, iteration, other_params) -> None:
        """
        Class constructor.

        Parameters
        ----------
        G : torch.nn.Module
            The generator model.
        D : torch.nn.Module
            The discriminator model.
        G_loss : float
            The generator loss.
        D_loss : float
            The discriminator loss.
        epoch : int
            The epoch number.
        iteration : int
            The iteration number.
        other_params : dict
            Other parameters to save with the model.
        """
        self.G = G
        self.D = D
        self.G_loss = G_loss
        self.D_loss = D_loss
        self.epoch = epoch
        self.iteration = iteration
        self.params = other_params

    # TODO use these to compare models; what other metrics should we use?
    def __lt__(self, other):
        return self.G_loss < other.G_loss

    def __gt__(self, other):
        return self.G_loss > other.G_loss


class HallOfFame:
    """
    Records the best models during training.
    """
    def __init__(self, num_models: int = 5) -> None:
        """
        Class constructor.

        Parameters
        ----------
        num_models : int (default: 5)
            The number of models to keep in the hall of fame.
        metric : str (default: 'loss')
            The metric to use for determining the best models.
        maximize : bool (default: False)
            Whether to maximize or minimize the metric.
        """
        self.num_models = num_models
        self.models = []

    def update(self, G, D, G_loss, D_loss, epoch, iteration, other_params=None) -> None:
        """
        Update the hall of fame with a new model.

        Parameters
        ----------
        G : torch.nn.Module
            The generator model.
        D : torch.nn.Module
            The discriminator model.
        G_loss : float
            The generator loss.
        D_loss : float
            The discriminator loss.
        epoch : int
            The epoch number.
        iteration : int
            The iteration number.
        other_params : dict (default: None)
            Other parameters to save with the model.
        """
        snapshot = SnapShot(G, D, G_loss, D_loss, epoch, iteration, other_params)
        self.models.append(snapshot)
        self.models.sort(reverse=True)
        if len(self.models) > self.num_models:
            self.models = self.models[:self.num_models]

    def get_best_model(self) -> SnapShot:
        return self.models[0]
