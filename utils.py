from tqdm import tqdm  # type: ignore


class ProgressBar(tqdm):  # type: ignore
    """
    Source: https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    """

    def update_to(self, b: int = 1, bsize: int = 1, tsize: None =None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)