import numpy as np
import pandas as pd

class EwmWrapper():
    """
    Inverse of exponentially weighted moving average.

    Parameters
    ----------
    com : float, optional
        Specify decay in terms of center of mass,
        :math:`alpha = 1 / (1 + com),text{ for } com geq 0`.

    span : float, optional
        Specify decay in terms of span,
        :math:`alpha = 2 / (span + 1),text{ for } span geq 1`.

    halflife : float, optional
        Specify decay in terms of half-life,
        :math:`alpha = 1 - exp(log(0.5) / halflife),text{for} halflife > 0`.

    alpha : float, optional
        Specify smoothing factor :math:`alpha` directly,
        :math:`0 < alpha leq 1`.

    min_periods : int, default 0
        Minimum number of observations in window required to have a value
        (otherwise result is NA).

    adjust : bool, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings
        (viewing EWMA as a moving average).

    ignore_na : bool, default False
        Ignore missing values when calculating weights;
        specify True to reproduce pre-0.15.0 behavior.

    axis : {0 or 'index', 1 or 'columns'}, default 0
        The axis to use. The value 0 identifies the rows, and 1
        identifies the columns.

    Attributes
    -----------
    ewm : Exponential weighted function for the dataframe

    ewma : Exponential weighted moving average for the
        dataframe calculated from ewm
    """

    def __init__(self, df, com=None, span=None, halflife=None, alpha=None,
                 min_periods=0, adjust=True, ignore_na=False, axis=0):
        self.com = com
        self.span = span
        self.halflife = halflife
        self.min_periods = min_periods
        self.adjust = adjust
        self.ignore_na = ignore_na
        self.axis = axis
        self.df = df
        self.ewma = None

        # Calculate the alpha value and handle errors
        self.alpha = self.get_alpha(alpha)

        # Create the ewm object
        self.ewm = df.ewm(com=com, span=span, halflife=halflife, alpha=alpha,
                          min_periods=min_periods, adjust=adjust, ignore_na=ignore_na, axis=axis)


    def calc_ewma(self):
        """Calculate the exponential weighted moving average
           on the ewm object

        Returns
        --------
        ewma : ndarray
            exponential weighted moving average of df
        """

        self.ewma = self.ewm.mean()
        return self.ewma


    def get_alpha(self, alpha):
        """Validate the input center of mass variables
           calculate the decay value (alpha)

        Parameters
        ------------
        alpha: Input alpha (decay value)

        Returns
        -------
        alpha : Validated and computed alpha based on inputs.
        """

        valid_count = sum(x is not None for x in [self.com, self.span, self.halflife, alpha])

        if valid_count > 1:
            raise ValueError("com, span, halflife, and alpha are mutually exclusive")

        # Convert to alpha
        # domain checks ensure 0 < alpha <= 1
        if alpha is not None:
            if alpha <= 0 or alpha > 1:
                raise ValueError("alpha must satisfy: 0 < alpha <= 1")
            return alpha

        elif self.com is not None:
            if self.com < 0:
                raise ValueError("com must satisfy: com >= 0")
            return 1 / (1 + self.com)

        elif self.span is not None:
            if self.span < 1:
                raise ValueError("span must satisfy: span >= 1")
            return 2 / (self.span + 1)

        elif self.halflife is not None:
            if self.halflife <= 0:
                raise ValueError("halflife must satisfy: halflife > 0")
            return (1 - np.exp(np.log(0.5) / self.halflife))

        else:
            raise ValueError("Must pass one of com, span, halflife, or alpha")


    def inverse_ewma(self):
        """Find the original df from the ewma_df

        Returns
        --------
        inversed_ewma: ndarray
        """

        if self.ewma is None:
            raise ValueError("EWMA was not calculated for te given df")

        ewma_values = self.ewma.values
        columns = self.ewma.columns

        # Calculate each value using the formulae
        def apply_inverse(ewma_values):
            """Apply the inverse ewma for a 1d array

            Parameters
            -----------
            ewma_values: 1d array of ewma values (columns of df)

            Returns
            --------
            inversed_ewma : inverse ewma of the given column of df
            """

            inversed_ewma = np.zeros(ewma_values.shape)

            # The x0 value is the same as y0
            inversed_ewma[0] = ewma_values[0]

            # Using the formulae to calculate individual values
            if self.adjust:
                for i, val in enumerate(ewma_values[1:]):
                    inversed_ewma[i + 1] = val + sum([pow((1 - self.alpha), ind + 1) * (val - xtm1) for ind, xtm1 in
                                                  enumerate(reversed(inversed_ewma[:i + 1]))])

            else:
                for i, val in enumerate(ewma_values[1:]):
                    inversed_ewma[i + 1] = (val - (1 - self.alpha) * ewma_values[i]) / self.alpha
            return inversed_ewma

        inversed_ewma = np.apply_along_axis(apply_inverse, 0, ewma_values)

        self.inversed_df = pd.DataFrame(inversed_ewma, columns=columns)
        return self.inversed_df
