class ConfidenceInterval:
    """
    A class to store confidence interval values.
    """

    def __init__(self, lower: float, upper: float) -> None:
        self.lower: float = lower
        self.upper: float = upper
        self.ci: tuple[float, float] = (lower, upper)


class ApproximatedStandardError:
    """
    A class to store approximated standard error values.
    """

    # def __init__(self) -> None:
    #     self.true_se: float = 0.0
    #     self.morris: float = 0.0
    #     self.hedges: float = 0.0
    #     self.hedges_olkin: float = 0.0
    #     self.mle: float = 0.0
    #     self.large_n: float = 0.0
    #     self.hunter_and_schmidt: float = 0.0

    def __init__(
        self,
        true_se: float,
        morris: float,
        hedges: float,
        hedges_olkin: float,
        mle: float,
        large_n: float,
        hunter_and_schmidt: float,
    ) -> None:
        self.true_se: float = true_se
        self.morris: float = morris
        self.hedges: float = hedges
        self.hedges_olkin: float = hedges_olkin
        self.mle: float = mle
        self.large_n: float = large_n
        self.hunter_and_schmidt: float = hunter_and_schmidt


class EffectSize:
    """
    A class to store effect size values.
    """

    def __init__(self, value, ci_lower, ci_upper, standard_error) -> None:
        self.effect_size_name: str = "Effect Size"
        self.value: float = value
        self.ci: ConfidenceInterval = ConfidenceInterval(ci_lower, ci_upper)
        self.standard_error: float = standard_error
        self.statistical_line: str = ""

        self.standardizer: float | None = None
        self.non_central_ci: ConfidenceInterval | None = None
        self.pivotal_ci: ConfidenceInterval | None = None
        self.approximated_standard_error: ApproximatedStandardError | None = None

    def update_statistical_line(self) -> None:
        """
        Update the statistical line with the current values.
        """
        self.statistical_line = (
            f"{self.effect_size_name}: {self.value} (CI: {self.ci.lower}, {self.ci.upper}) "
            f"SE: {self.standard_error}"
        )

        if self.non_central_ci is not None:
            self.statistical_line += f" Non-Central CI: {self.non_central_ci.lower}, {self.non_central_ci.upper}"

        if self.pivotal_ci is not None:
            self.statistical_line += (
                f" Pivotal CI: {self.pivotal_ci.lower}, {self.pivotal_ci.upper}"
            )

    def update_non_central_ci(
        self, non_central_ci_lower: float, non_central_ci_upper: float
    ) -> None:
        """
        Update the non-central confidence interval.
        """
        self.non_central_ci = ConfidenceInterval(
            non_central_ci_lower, non_central_ci_upper
        )
        self.update_statistical_line()

    def update_pivotal_ci(
        self, pivotal_ci_lower: float, pivotal_ci_upper: float
    ) -> None:
        """
        Update the pivotal confidence interval.
        """
        self.pivotal_ci = ConfidenceInterval(pivotal_ci_lower, pivotal_ci_upper)
        self.update_statistical_line()


class CohenD(EffectSize):
    """
    A class to store Cohen's d effect size values.
    """

    def __init__(self, value, ci_lower, ci_upper, standard_error) -> None:
        super().__init__(value, ci_lower, ci_upper, standard_error)
        self.effect_size_name: str = "Cohen's d"
        self.update_statistical_line()


class HedgesG(EffectSize):
    """
    A class to store Hedges' g effect size values.
    """

    def __init__(self, value, ci_lower, ci_upper, standard_error) -> None:
        super().__init__(value, ci_lower, ci_upper, standard_error)
        self.effect_size_name: str = "Hedges' g"
        self.update_statistical_line()
