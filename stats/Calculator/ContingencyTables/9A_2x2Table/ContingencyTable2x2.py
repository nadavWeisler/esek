
import scipy.stats as st
import numpy as np

# The corrected version of the terachoric correlation is still missing

class Contingency_2X2():
    @staticmethod
    def Binary_2x2_measures(params: dict) -> dict: 
        
        Contingency_Table = params["Contingency Table"]
        Confidence_Level_Percentages = params["Confidence Interval"]

        # Preperations
        confidence_level = Confidence_Level_Percentages / 100   
        a = Contingency_Table[0,0]
        b = Contingency_Table[0,1]
        c = Contingency_Table[1,0]
        d = Contingency_Table[1,1]
        sample_size = a+b+c+d
        
        # Effect Sizes

        # 1. Tetrachoric Correlation
        Odds_Ratio = a*d/ (b*c)
        Tetrachoric_Correlation = np.cos(3.14159265359 / (1 + np.sqrt(Odds_Ratio))) 
        Tetrachoric_Basic_Approximation = (Odds_Ratio**0.74 - 1) / (Odds_Ratio**0.74 + 1)
        
        # 2. Corrected Tetrachoric Correlation
        r1 = (a + b + 1) / (sample_size + 2)
        r2 = (c + d + 1) / (sample_size + 2)
        c1 = (a + b + 1) / (sample_size + 2)
        c2 = (c + d + 1) / (sample_size + 2)
        Minimal_Probability = min(c1, c2, r1, r2)
        Correction = (1 - abs(r1 - c1) / 5 - (0.5 - Minimal_Probability)**2) / 2
        Corrected_Odds_Ratio = (a + 0.5)*(b + 0.5)/((c + 0.5)*(d + 0.5))
        Corrected_Tetrachoric_Correlation = np.cos(3.14159/ (1 + Corrected_Odds_Ratio**Correction))
        
        # Corrected Tetrachoric Correlaiton and Its Variance based on the odds ratio variance (Bonnet, 2005)

        # 3. Chi Square and phi
        phi = (a*d - b*c) / np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
        Cramer = phi
        chi_square_phi = phi**2*sample_size
        p_value = st.chi2.sf((abs(chi_square_phi)), 1)


        # 4. Bias Corrected Chi Sqare (Bergsma, 2013)
        Corrected_levels_Number = 2 - 1/(sample_size-1)
        Bias_Corrected_Cramer = np.sqrt((phi**2/(Corrected_levels_Number-1))) 
        p_value_Bias_Corrected_phi = st.chi2.sf((abs(Bias_Corrected_Cramer)), 1)
        

        # 5. Maximum Corrected Phi (See richard liu 1980)
        max_phi1 = np.sqrt( (  (a+b)/(c+d)) * ((d+b) / (c+a)))
        max_phi2 = np.sqrt( (  (a+c)/(b+d)) * ((a+b) / (c+d)))  
        max_phi = max_phi1 if c + d > b + a else max_phi2
        Max_Corrected_Phi = phi/max_phi
        chi_square_max_corrected_phi = Max_Corrected_Phi**2*sample_size
        p_value_max_Corrected_phi = st.chi2.sf((abs(chi_square_max_corrected_phi)), 1)

        # 6. Wallis' Swing d 
        Wallis_Swing_d_rows_indpendnt =  a/(a+c) - b/(b+d)
        Wallis_Perecentage_Swing_rows_independent =  Wallis_Swing_d_rows_indpendnt /  (a/(a+c))
        Estimate = (a+b)/sample_size
        Standard_Error_rows_independent = np.sqrt(( (Estimate*(1-Estimate)) * (1/(a+c)+(1/(b+d)))))
        Wallis_Swing_d_coloumns_indpendnt =  a/(a+b) - c/(c+d)
        Wallis_Perecentage_Swing_coloumns_independent =  Wallis_Swing_d_coloumns_indpendnt /  (a/(a+b))
        Estimate = (a+c)/sample_size
        Standard_Error_coloumns_independent = np.sqrt(( (Estimate*(1-Estimate)) * (1/(a+b)+(1/(c+d)))))

        # 7. More Measures
        Chambers_R = (((Odds_Ratio + 1)/(Odds_Ratio - 1))  -  ((2*Odds_Ratio * np.log(Odds_Ratio)) / (Odds_Ratio - 1)**2))

        
        # Confidecne Intervals
        
        # 1. Tetrachoric Correlation 
        z_crit = st.norm.ppf(confidence_level + ((1 - confidence_level) / 2))
        Standard_Error_Odds_Ratio = np.sqrt(1/a + 1/b + 1/c + 1/d)
        Standard_Error_Odds_Ratio_Corrected = np.sqrt(1 / (a + 0.5) + 1 / (b + 0.5) + 1 / (c + 0.5) + 1 / (d + 0.5))
        K = (3.14159 * 0.5 * Odds_Ratio**0.5) * np.sin(3.14159 / (1 + Odds_Ratio**0.5)) / (1 + Odds_Ratio**0.5)**2
        Corrected_K = (3.14159 * Correction * Corrected_Odds_Ratio**Correction) * np.sin(3.14159/(1 + Corrected_Odds_Ratio**Correction)) / (1 + Corrected_Odds_Ratio**Correction)**2
        Standard_Error_Tetrachoric = K * Standard_Error_Odds_Ratio_Corrected
        Standard_Error_Tetrachoric_Corrected = Corrected_K * Standard_Error_Odds_Ratio_Corrected


        # Phi Square

        # calculation of the Variance of phi (from Bishop et al's., book) (approximation for large samples)
        p1_plus = (a +b)/sample_size # Marginal_Probability_Row1
        p2_plus = (c +d)/sample_size # Marginal_Probability_Row2
        pplus_1 = (a +c)/sample_size # Marginal_Probability_Coloumn1
        pplus_2 = (b +d)/sample_size # Marginal_Probability_Coloumn2
        Probabilities_Product = (p1_plus*p2_plus*pplus_1*pplus_2)
        term1 =  phi + 0.5 * phi**3
        term2 = ((p1_plus - p2_plus) * (pplus_1 - pplus_2)) / np.sqrt(Probabilities_Product)
        term3 = 0.75*phi**2*(((p1_plus - p2_plus)**2 / (p1_plus * p2_plus)) + ((pplus_1 - pplus_2)**2 / (pplus_1 * pplus_2)))
        Variance = (1/sample_size) * (1 - phi**2 + term1 * term2 - term3)





        # Wallis Swing D CI's
        LowerCi_swing_d_Wallis = Wallis_Swing_d_coloumns_indpendnt - Standard_Error_coloumns_independent*z_crit
        UpperCi_swing_d_Wallis = Wallis_Swing_d_coloumns_indpendnt + Standard_Error_coloumns_independent*z_crit
        LowerCi_Percentage_swing_d_Wallis = LowerCi_swing_d_Wallis / (a/(a+b))
        UpperCi_Percentage_swing_d_Wallis = UpperCi_swing_d_Wallis / (a/(a+b))
        LowerCi_swing_d_Wallis_rows = Wallis_Swing_d_coloumns_indpendnt - Standard_Error_coloumns_independent*z_crit
        UpperCi_swing_d_Wallis_rows = Wallis_Swing_d_coloumns_indpendnt + Standard_Error_coloumns_independent*z_crit
        LowerCi_Percentage_swing_rows_d_Wallis = LowerCi_swing_d_Wallis_rows / (a/(a+c))
        UpperCi_Percentage_swing_rows_d_Wallis = UpperCi_swing_d_Wallis_rows / (a/(a+c))



        results = {}

        results["Odds Ratio"] = round(Odds_Ratio, 4)
        results["Chambers r"] = round(Chambers_R, 4)
        
        # Tetrachoric Correlation
        results["Tetrachoric Correlation"] = round(Tetrachoric_Correlation, 4)
        results["Tetrachoric Approximation"] = round(Tetrachoric_Basic_Approximation, 4)
        results["Tetrachoric Corrected"] = round(Corrected_Tetrachoric_Correlation, 4)
        results["Standard_Error_Tetrachoric"] = round(Standard_Error_Tetrachoric, 4)
        results["Standard_Error_Tetrachoric_Corrected"] = round(Standard_Error_Tetrachoric_Corrected, 4)

        # Phi and max phi
        results["Phi / Cohen's w"] = round(phi, 4)
        results["Phi chi square statistic"] = round(chi_square_phi, 4)
        results["Phi p_value"] = np.round(p_value, 4)
        results["maxphi"] = round(max_phi, 4)
        results["Max_Corrected_Phi (Liu, 1980)"] = round(Max_Corrected_Phi, 4)
        results["Phi max_Corrected_p_value"] = np.round(p_value_max_Corrected_phi, 4)
        results["Phi Variance"] = round(Variance, 7)
        results["chi square phi"] = round(chi_square_phi, 5)
        results["chisquare max corrected phi"] = round(chi_square_max_corrected_phi, 5)

        # Cramer V (Which is equal to phi) and it's biased corrected version
        results["Cramer's V"] = round(Cramer, 4)
        results["Bias Corrected Cramer (Bergsma, 2013)"] = round(Bias_Corrected_Cramer, 4) # Bergsma, 2013

        # Wallis Measures of effect size
        results["Wallis Swing D (coloumns independent)"] = round(Wallis_Swing_d_coloumns_indpendnt, 4)
        results["Wallis Swing D Standard Error"] = round(Standard_Error_coloumns_independent, 5)
        results["Wallis Percentage Swing (coloumns independent)"] = round(Wallis_Perecentage_Swing_coloumns_independent, 4)
        results["CI_swing_d_Wallis"] = f"({round(LowerCi_swing_d_Wallis, 4)}, {round(UpperCi_swing_d_Wallis, 4)})"
        results["CI_Percentage_swing_d_Wallis"] = f"({round(LowerCi_Percentage_swing_d_Wallis, 4)}, {round(UpperCi_Percentage_swing_d_Wallis, 4)})"

        results["Wallis Swing D (rows independent)"] = round(Wallis_Swing_d_rows_indpendnt, 4)
        results["Wallis Swing D Standard Error (rows)"] = round(Standard_Error_rows_independent, 5)
        results["Wallis Percentage Swing (rows independent)"] = round(Wallis_Perecentage_Swing_rows_independent, 4)
        results["CI_swing_d_Wallis (rows)"] = f"({round(LowerCi_swing_d_Wallis_rows, 4)}, {round(UpperCi_swing_d_Wallis_rows, 4)})"
        results["CI_Percentage_swing_d_Wallis (rows)"] = f"({round(LowerCi_Percentage_swing_rows_d_Wallis, 4)}, {round(UpperCi_Percentage_swing_rows_d_Wallis, 4)})"

        #result_str = "\n".join([f"{key}: {value}" for key, value in results.items()])
        #return result_str

        return results


    #Binary_2x2_measures(a,b,c,d)







