import pingouin as pg
import numpy as np
from scipy.stats import norm

###############################
# Partial_Correlation_Pearson #
###############################
class Partial_Correlation_Pearson():
    @staticmethod
    def partial_correlation_from_data(params: dict) -> dict:

        data = params["Data"]
        Confidence_Level_Percentages = params["Confidence Level"]

        confidence_level = Confidence_Level_Percentages / 100
        df = data.drop(columns=['independent_variable', 'dependnent_variable'], errors='ignore')
        covariance_variables = df.columns.tolist()

        # Partial Correlation
        partial_correlation_output = pg.partial_corr(data, 'independent_variable', 'dependnent_variable', covar=covariance_variables)
        Sample_Size = partial_correlation_output.values[0,0]
        partial_correlation = partial_correlation_output.values[0,1]
        p_value_partial_correlation = partial_correlation_output.values[0,3]

        # Confidence Intervals
        zcrit = norm.ppf(1 - (1 - confidence_level) / 2)
        LowerCi_Partial = np.arctan(np.arctanh(partial_correlation) - zcrit *(1/(np.sqrt(Sample_Size - 3))))
        UpperCi_Partial = np.arctan(np.arctanh(partial_correlation) + zcrit *(1/(np.sqrt(Sample_Size - 3))))

        # Semi Partial Correlation
        semi_partial_correlation_output = pg.partial_corr(data, 'independent_variable', 'dependnent_variable', x_covar = covariance_variables)
        semi_partial_correlation = semi_partial_correlation_output.values[0,1]
        p_value_semi_partial_correlation = semi_partial_correlation_output.values[0,3]
        LowerCi_Semi_Partial = np.arctan(np.arctanh(semi_partial_correlation) - zcrit *(1/(np.sqrt(Sample_Size - 3))))
        UpperCi_Semi_Partial = np.arctan(np.arctanh(semi_partial_correlation) + zcrit *(1/(np.sqrt(Sample_Size - 3))))


        results = {}
        results['Sample Size'] = Sample_Size
        results['Partial Correlation'] = partial_correlation
        results['Confidence Intervals Partial Correlation'] = [LowerCi_Partial, UpperCi_Partial]
        results['p-value partial correlation '] = p_value_partial_correlation
        results['Semi Partial Correlation'] = semi_partial_correlation
        results['Confidence Intervals Semi Partial Correlation'] = [LowerCi_Semi_Partial, UpperCi_Semi_Partial]
        results['p-value Semi partial correlation '] = p_value_semi_partial_correlation
        formatted_p_value_partial = "{:.3f}".format(p_value_partial_correlation).lstrip('0') if p_value_partial_correlation >= 0.001 else "\033[3mp\033[0m < .001"
        formatted_p_value_semi_partial = "{:.3f}".format(p_value_semi_partial_correlation).lstrip('0') if p_value_semi_partial_correlation >= 0.001 else "\033[3mp\033[0m < .001"
        results["Statistical Line Partial Correlation"] = "\033[3mr\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((Sample_Size - 2), ('-' if np.round(partial_correlation,3) < 0 else ''), str(np.abs(np.round(partial_correlation,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_partial_correlation >= 0.001 else '', formatted_p_value_partial, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(LowerCi_Partial,3) < 0 else ''), str(np.abs(np.round(LowerCi_Partial,3))).lstrip('0').rstrip(''), ('-' if np.round(UpperCi_Partial,3) < 0 else ''), str(np.abs(np.round(UpperCi_Partial,3))).lstrip('0').rstrip(''))
        results["Statistical Line Semi Partial Correlation"] = "\033[3mr\033[0m({}) = {}{}, {}{}, {}{}% CI [{}{}, {}{}]".format((Sample_Size - 2), ('-' if np.round(partial_correlation,3) < 0 else ''), str(np.abs(np.round(partial_correlation,3))).lstrip('0').rstrip(''), '\033[3mp = \033[0m' if p_value_semi_partial_correlation >= 0.001 else '', formatted_p_value_semi_partial, int(confidence_level*100) if confidence_level.is_integer() else '{:.1f}'.format(confidence_level*100).rstrip('0').rstrip('.'), '' if confidence_level.is_integer() else '', ('-' if np.round(LowerCi_Semi_Partial,3) < 0 else ''), str(np.abs(np.round(LowerCi_Semi_Partial,3))).lstrip('0').rstrip(''), ('-' if np.round(UpperCi_Semi_Partial,3) < 0 else ''), str(np.abs(np.round(UpperCi_Semi_Partial,3))).lstrip('0').rstrip(''))


        return results



