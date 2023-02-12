
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as sm_diagnostics
import statsmodels.stats as sm_stats
import matplotlib.pyplot as plt
import scipy as scipy
import pandas as pd
import seaborn as sns
import os

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:05:37 2019
@author: Rory Boyle rorytboyle@gmail.com github.com/rorytboyle
"""

def linear_reg(y, X, names):
    """
    Runs a linear regression using statsmodels, extracts results from the
    model. Returns a list of results and the OLS results object.
    :param y: outcome variable
    :param X: predictor variable(s)
    :param names: names of predictor variables
    :return results: list of extracted stats/results from statsmodels OLS object
    :return model: OLS results object
    """
    # run regression - add column of 1s to X to serve as intercept
    model = sm.OLS(y, sm.add_constant(X)).fit()
     
    # extract results from statsmodel OLS object
    results = [names, model.nobs, model.df_resid, model.df_model,
               model.rsquared, model.fvalue, model.f_pvalue, model.ssr,
               model.centered_tss, model.mse_model, model.mse_resid,
               model.mse_total]

    # deep copy names and add constant - otherwise results list will contain
    # multiple repetitions of constant (due to below loop)
    namesCopy = names[:]
    namesCopy.insert(0, 'constant')
    
    # create dicts with name of each parameter in model (i.e. predictor
    # variables) and the beta coefficient andp-value
    coeffs = {}
    p_values = {}
    for ix, coeff in enumerate(model.params):
        coeffs[namesCopy[ix]] = coeff
        p_values[namesCopy[ix]] = model.pvalues[ix]
        
    results.append(coeffs)
    results.append(p_values)
    
    return results, model

def calculate_change_stats(model_stats):
    """
    Calculates r-squared change, f change, and p-value of f change for
    hierarchical regression results.
    
    f change is calculated using the formula:
    (r_squared change from Step 1 to Step 2 / no. predictors added in Step 2) / 
    (1 - step 2 r_squared) / (no. observations - no. predictors - 1)
    https://www.researchgate.net/post/What_is_a_significant_f_change_value_in_a_hierarchical_multiple_regression
        
    p-value of f change calculated using the formula:
    f with (num predictors added, n - k - 1) ==> n-k-1 = Residual df for Step 2
    https://stackoverflow.com/questions/39813470/f-test-with-python-finding-the-critical-value
    
    :param model_stats: description of parameter x
    :return: list containing r-squared change value, f change value, and
             p-value for f change
    
    """
    # get number of steps 
    num_steps = model_stats['step'].max()
    
    # calculate r-square change (r-sq of current step minus r-sq of previous step)
    r_sq_change = [model_stats.iloc[step+1]['r-sq'] -
                   model_stats.iloc[step]['r-sq'] for step in
                   range(0, num_steps-1)]
    
    # calculate f change - formula from here: 
    f_change = []
    for step in range(0, num_steps-1):
        # numerator of f change formula
        # (r_sq change / number of predictors added)
        f_change_numerator = r_sq_change[step] / (len(model_stats.iloc[step+1]['predictors'])
                                                  - len(model_stats.iloc[step]['predictors']))
        # denominator of f change formula
        # (1 - step2 r_sq) / (num obs - number of predictors - 1)
        f_change_denominator = ((1 - model_stats.iloc[step+1]['r-sq']) /
                                model_stats.iloc[step+1]['df_resid'])
        # compute f change
        f_change.append(f_change_numerator / f_change_denominator)
        
    # calculate pvalue of f change
    f_change_pval = [scipy.stats.f.sf(f_change[step], 1,
                                      model_stats.iloc[step+1]['df_resid'])
                     for step in range(0, num_steps-1)]
    
    return [r_sq_change, f_change, f_change_pval]

def hierarchical_regression(y, X, names, saveFolder):
    """
    Runs hierarchical linear regressions predicting y from X. Uses statsmodels
    OLS to run linear regression for each step. Returns results of regression
    in each step as well as r-squared change, f change, and p-value of f change
    for the change from step 1 to step 2, step 2 to step 3, and so on.
    
    The number of lists contained within names specifies the number of steps of
    hierarchical regressions. If names contains two nested lists of strings,
    e.g. if names = [[variable 1], [variable 1, variable 2]], then a two-step
    hierarchical regression will be conducted.
    
    :param y: outcome variable (1d array/series)
    :param X: nested lists with each list containing predictor variables for
              each step - if running a two step regression, X should contain
              two lists, which may contain a series or dataframe.
              If Step 1 contains a variable "height", and Step 2 contains
              "height" and "weight", then X should be:
              [[height], [height, weight]]
    :param names: nested lists with each list containing names of predictor
              variables for each step. names should be structured as above.
    :param saveFolder: full path for folder in which to save model results and/
              diagnostics info
    :return: model_stats - a df (rows = number of steps * cols = 18)
    with following info for each step:
        step = step number
        x = predictor names
        num_obs = number of observations in model
        df_resid = df of residuals
        df_mod = df of model
        r-sq = r-squared
        f = f-value
        f_pval = p-value
        sse = sum of squares of errors
        ssto = total sum of squares
        mse_mod = mean squared error of model
        mse_resid =  mean square error of residuals
        mse_total = total mean square error
        beta_coeff = coefficient values for intercept and predictors
        p_values = p-values for intercept and predictors
        r-sq_change = r-squared change for model (Step 2 r-sq - Step 1 r-sq)
        f_change = f change for model (Step 2 f - Step 1 f)
        f_change_pval = p-value of f-change of model
    :return reg_models: - a nested list containing the step name of each model
    and the OLS model object 
    """
          
    # Parse input
    # Make sure names and X are the same length, return error if not
    
    # Loop through steps and run regressions for each step
    results =[]
    reg_models = []
    for ix, currentX in enumerate(X):
        # run regression - ### CHANGED 19/08 TO ADD IN DIAGNOSTICS
        currentStepResults, currentStepModel = linear_reg(y, currentX, names[ix])
        currentStepResults.insert(0, ix+1)  # add step number to results

        saveto = saveFolder + r'\step' + str(ix+1)
        modelSave = saveto + "model.pickle" # THIS LINE HASNT BEEN TESTED
        currentStepModel.save(modelSave) # THIS LINE HASNT BEEN TESTED
        
        # run regression diagnostics
        assumptionsToCheck = regression_diagnostics(
                currentStepModel, currentStepResults, y, currentX, saveto)
        currentStepResults.append(assumptionsToCheck)
        
        results.append(currentStepResults)
        # add model to list of models along with step number
        reg_models.append(['Step ' + str(ix+1), currentStepModel])
        
    # add results to model_stats dataframe
    model_stats = pd.DataFrame(results)
    model_stats.columns = ['step', 'predictors', 'num_obs', 'df_resid',
                           'df_mod', 'r-sq', 'f', 'f_pval', 'sse', 'ssto',
                           'mse_mod', 'mse_resid', 'mse_total', 'beta_coeff',
                           'p_values', 'assumptionsToCheck']
    
    # calculate r-sq change, f change, p-value of f change
    change_results = calculate_change_stats(model_stats)
    
    # append step number to change results
    step_nums = [x+1 for x in [*range(1, len(change_results[0])+1)]]
    change_results.insert(0, step_nums)
    
    # add change results to change_stats dataframe
    change_stats = pd.DataFrame(change_results).transpose()
    change_stats.columns = ['step', 'r-sq_change', 'f_change', 'f_change_pval'] 
    
    # merge model_stats and change_stats
    model_stats = pd.merge(model_stats, change_stats, on='step', how='outer')
        
    return model_stats, reg_models


# https://www.statsmodels.org/dev/examples/notebooks/generated/regression_plots.html
# https://towardsdatascience.com/verifying-the-assumptions-of-linear-regression-in-python-and-r-f4cd2907d4c0

def regression_diagnostics(model, result, y, X, saveto):
    """
    Runs formal diagnostic tests for linear regression and creates plots for
    further inspection. Outputs a summary text file listing the failed
    diagnostic tests and a list of assumptions that require further inspection.
    Assumption tested               Diagnostic Test(s) & Plots used
    1. Independence of Residuals    Durbin Watson Test
    
    2. Linearity                    Pearson's Correlations for DV and each IV
                                    Harvey-Collier Multiplier test
                                    Rainbow Test
                                    Plot: Studentised Residuals vs Fitted Values
                                    Plot: Partial Regression Plots
                                    
    3. Homoscedasticity             Breusch Pagan Test
                                    F-test
                                    Goldfeld Quandt Test
                                    Plot: Studentised Residuals vs Fitted Values
                                    
    4. Multicollinearity            Pairwise Correlations between DVs
                                    Variance Inflation Factor
                                    
    5. Outliers/Influence           Standardised Residuals (> -3 & < +3)
                                    Cook's Distance
                                    Plot: Boxplot of Standardised Residuals
                                    Plot: Influence Plot with Cook's Distance
                                    
    6. Normality                    Mean of Residuals (approx = 0)
                                    Shapiro-Wilk Test
                                    Plot: Normal QQ Plot of Residuals
    :param model: regression.linear_model.RegressionResultsWrapper
                  from statsmodels.OLS
    :param result: Series containing extracted results from
                  linear regression. One row of results df
                  returned by hierarchical_regression()
    :param y: outcome variable
    :param X: predictor variable(s)
    :param saveto: folder specifying dir to save results and
                  plots
    :return assumptionsToCheck: list of assumptions that require further
                                inspection
    """
    # get resid vals
    influence_df = sm_diagnostics.OLSInfluence(model).summary_frame()
    # create dict to store diagnostics test info
    diagnostics = {}
    # create dict to link diagnostic tests to assumptions
    assumptionTests = {}
    # create dict with formal names of diagnostic tests - for printing warnings
    formalNames = {}
    # create folder
    try:
        os.makedirs(saveto)
    except:
        pass
    # get step number
    step = saveto.split("\\")[-1]

# ASSUMPTION 1 - INDEPENDENCE OF RESIDUALS
    # Durbin-Watson stat (no autocorrelation)
    diagnostics['durbin_watson_stat'] = sm_stats.stattools.durbin_watson(
            model.resid, axis=0)
    # Acceptable Durbin-Watson values = 1.5 to 2.5
    if diagnostics['durbin_watson_stat'] >= 1.5 and diagnostics[
            'durbin_watson_stat'] <= 2.5:
        diagnostics['durbin_watson_passed'] = 'Yes'
    else:
        diagnostics['durbin_watson_passed'] = 'No'
    # link test to assumption
    assumptionTests['durbin_watson_passed'] = 'Independence of Residuals'
    formalNames['durbin_watson_passed'] = 'Durbin-Watson Test'

# ASSUMPTION 2 - LINEARITY
    # a) linearity between DV and each IV - Pearson's r
    if len(X.columns) > 1:  # run code if there are multiple predictors
        # Debugged 30/09/2019 originally used just y in correlation function
        # y.iloc[:, 0] needed to call function between two series to avoid error
        correlations = [scipy.stats.pearsonr(X[var], y.iloc[:, 0])
                        for var in X.columns]
        for ix, corr in enumerate(correlations):
            xName = 'IV_' + X.columns[ix] + '_pearson_'
            diagnostics[xName + 'r'] = corr[0]
            diagnostics[xName + 'p'] = corr[1]
    else:  # run code if only 1 predictor
        correlations = scipy.stats.pearsonr(X, y)

    # search through dict for keys with pearson_p and assign yes to passed var
    # if all p's < 0.05
    nonSigLinearIV_toDV = 0  # flag
    nonSigLinearVars = []
    for key in diagnostics:
        if key[-9:] == 'pearson_p':
            if diagnostics[key] > 0.05:
                nonSigLinearIV_toDV += 1
                nonSigLinearVars.append(key)

    if nonSigLinearIV_toDV == 0:
        diagnostics['linear_DVandIVs_passed'] = 'Yes'
    else:
        diagnostics['linear_DVandIVs_passed'] = 'No:' + ', '.join(
                nonSigLinearVars)
    # link test to assumption
    assumptionTests['linear_DVandIVs_passed'] = 'Linearity'
    formalNames['linear_DVandIVs_passed'] = 'Non-sig. linear relationship between DV and each IV'

    # b) linearity between DV and IVs collectively
    # Harvey-Collier multiplier test for linearity -
    # null hypo = residuals (and thus the true model) are linear
    diagnostics[
            'harvey_collier_linearity'] = sm_stats.api.linear_harvey_collier(
            model)[1]

    if diagnostics['harvey_collier_linearity'] < 0.05:
        diagnostics['harvey_collier_linearity_passed'] = 'Yes'
    else:
        diagnostics['harvey_collier_linearity_passed'] = 'No'
    # link test to assumption
    assumptionTests['harvey_collier_linearity_passed'] = 'Linearity'
    formalNames['harvey_collier_linearity_passed'] = 'Harvey-Collier Multiplier Test'

    # rainbow test for linearity - null hypo = model has adequate linear fit
    diagnostics['rainbow_linearity'] = sm_stats.diagnostic.linear_rainbow(
            model)[1]

    if diagnostics['rainbow_linearity'] > 0.05:
        diagnostics['rainbow_linearity_passed'] = 'Yes'
    else:
        diagnostics['rainbow_linearity_passed'] = 'No'
    # link test to assumption
    assumptionTests['rainbow_linearity_passed'] = 'Linearity'
    formalNames['rainbow_linearity_passed'] = 'Rainbow Test'

# ASSUMPTION 3 - HOMOSCEDASTICITY
    # https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html
    breusch_pagan_test = sm_stats.diagnostic.het_breuschpagan(
            model.resid, model.model.exog)
    diagnostics['breusch_pagan_p'] = breusch_pagan_test[1]
    diagnostics['f_test_p'] = breusch_pagan_test[3]

    # if breusch pagan test is sig, then reject null hypo of homoscedasticity
    if diagnostics['breusch_pagan_p'] < .05:
        diagnostics['breusch_pagan_passed'] = 'No'
    else:
        diagnostics['breusch_pagan_passed'] = 'Yes'
    assumptionTests['breusch_pagan_passed'] = 'Homoscedasticity'
    formalNames['breusch_pagan_passed'] = 'Bruesch Pagan Test'

    # if f test is sig, then reject null hypo of homoscedasticity
    # f test more appropriate for for small or moderately large samples
    if diagnostics['f_test_p'] < .05:
        diagnostics['f_test_passed'] = 'No'
    else:
        diagnostics['f_test_passed'] = 'Yes'
    assumptionTests['f_test_passed'] = 'Homoscedasticity'
    formalNames['f_test_passed'] = 'F-test for residual variance'

    # Goldfeld Quandt test
    goldfeld_quandt_test = sm_stats.api.het_goldfeldquandt(
            model.resid, model.model.exog)
    diagnostics['goldfeld_quandt_p'] = goldfeld_quandt_test[1]
    # if goldfeld quandt test is sig, then reject null hypo of homoscedasticity
    if diagnostics['goldfeld_quandt_p'] < .05:
        diagnostics['goldfeld_quandt_passed'] = 'No'
    else:
        diagnostics['goldfeld_quandt_passed'] = 'Yes'
    assumptionTests['goldfeld_quandt_passed'] = 'Homoscedasticity'
    formalNames['goldfeld_quandt_passed'] = 'Goldfeld Quandt Test'

# ASSUMPTION 4 - MULTICOLLINEARITY
    # a) check pairwise correlations < 0.8
    if len(X.columns) > 1:  # run code if there are multiple predictors
        pairwise_corr = X.corr()
        pairwise_corr = pairwise_corr[pairwise_corr != 1]  # make diagonals=nan

        high_pairwise_corr = pairwise_corr[pairwise_corr >= 0.3]
        if high_pairwise_corr.isnull().all().all():
            diagnostics['high_pairwise_correlations_passed'] = 'Yes'
        else:
            diagnostics['high_pairwise_correlations_passed'] = 'No'
    else:  # run code if only 1 predictor
        diagnostics['high_pairwise_correlations_passed'] = 'Yes'

    # link test to assumption
    assumptionTests['high_pairwise_correlations_passed'] = 'Multicollinearity'
    formalNames['high_pairwise_correlations_passed'] = 'High Pairwise correlations'

    # b) Variance Inflation Factors < 10
    if len(X.columns) > 1:  # run code if there are multiple predictors
        vif = pd.DataFrame()
        vif['VIF'] = [sm_stats.outliers_influence.variance_inflation_factor(
                X.values, i) for i in range(X.shape[1])]
        vif['features'] = X.columns

        # if no predictors have vif > 5
        if ((vif['VIF'] < 5).all()):
            diagnostics['VIF_passed'] = 'Yes'
            diagnostics['VIF_predictorsFailed'] = []
        else:
            diagnostics['VIF_passed'] = 'No'
            # add predictor names to diagnostics
            diagnostics['VIF_predictorsFailed'] = vif[vif > 5].to_string(
                    index=False, header=False)

    else:  # run code if only 1 predictor
        diagnostics['VIF_passed'] = 'Yes'

    # link test to assumption
    assumptionTests['VIF_passed'] = 'Multicollinearity'
    formalNames['VIF_passed'] = 'High Variance Inflation Factor'

# ASSUMPTION 5 - OUTLIERS
    #  no outliers, high leverage pts, or highly influential pts
    # get index of outliers w/ std. resids above/below 3/-3
    highOutliers = influence_df[
            influence_df['standard_resid'] < -3].index.tolist()
    lowOutliers = influence_df[
            influence_df['standard_resid'] > 3].index.tolist()
    diagnostics['outlier_index'] = highOutliers + lowOutliers

    if not diagnostics['outlier_index']:
        diagnostics['outliers_passed'] = 'Yes'
    else:
        diagnostics['outliers_passed'] = 'No'
    # link test to assumption
    assumptionTests['outliers_passed'] = 'Outliers/Leverage/Influence'
    formalNames['outliers_passed'] = 'Extreme Standardised Residuals'

    # influence = Cook's Distance
    # https://www.researchgate.net/publication/2526564_A_Teaching_Note_on_Cook's_Distance_-_A_Guideline
    if len(X.columns) == 1:
        cooks_cutOff = 0.7  # cut off for 1 predictor = Cooks > 0.7 (n>15)
    elif X.shape[1] == 2:
        cooks_cutOff = 0.8  # cut off for 2 predictors = Cooks > 0.8 (n>15)
    elif X.shape[1] > 2:
        cooks_cutOff = 0.85  # cut off for >2 predictors = Cooks > 0.85 (n>15)

    diagnostics['influence_largeCooksD_index'] = influence_df[
            influence_df['cooks_d'] > cooks_cutOff].index.tolist()

    if not diagnostics['influence_largeCooksD_index']:
        diagnostics['influence_passed'] = 'Yes'
    else:
        diagnostics['influence_passed'] = 'No'
    # link test to assumption
    assumptionTests['influence_passed'] = 'Outliers/Leverage/Influence'
    formalNames['influence_passed'] = "Large Cook's Distance"

# ASSUMPTION 6 - NORMALITY
    #  normal distribution of residuals
    # check mean is 0 ( < 0.1 & > -0.1) and errors approx normally distributed
    diagnostics['meanOfResiduals'] = model.resid.mean()
    if diagnostics['meanOfResiduals'] < .1 and diagnostics[
            'meanOfResiduals'] > -.1:
        diagnostics['meanOfResiduals_passed'] = 'Yes'
    else:
        diagnostics['meanOfResiduals_passed'] = 'No'
    # link test to assumption
    assumptionTests['meanOfResiduals_passed'] = 'Normality'
    formalNames['meanOfResiduals_passed'] = "Mean of residuals not approx = 0"

    # Shapiro-Wilk test on residuals
    diagnostics['shapiroWilks_p'] = scipy.stats.shapiro(model.resid)[1]
    if diagnostics['shapiroWilks_p'] > 0.05:
        diagnostics['shapiroWilks_passed'] = 'Yes'
    else:
        diagnostics['shapiroWilks_passed'] = 'No'
    # link test to assumption
    assumptionTests['shapiroWilks_passed'] = 'Normality'
    formalNames['shapiroWilks_passed'] = 'Shapiro-Wilk Test'

# SUMMARISE DIAGNOSTIC TEST INFO
    # check whether diagnostic tests are passed. If all tests passed, then
    # print message telling user that model is ok. If any test failed, print
    # message telling user that model may not satisfy assumptions, check plots,
    # and investigate further.
    diagnostic_tests = 0
    diagnosticsPassed = 0
    violated = []

    print('\n\n\nDiagnostic summary for: ' + step)
    for key in diagnostics:
        if key[-6:] == 'passed':
            diagnostic_tests += 1
            if diagnostics[key] == 'Yes':
                diagnosticsPassed += 1
            else:
                # find which assumption diagnostic test referred to AND print
                # message telling user to investigate assumption further
                print('Diagnostic test (' + formalNames[key] +
                      ') failed for ' + assumptionTests[key])
                # add assumption to possible violations
                violated.append(assumptionTests[key])

    # summarise how many tests passed/failed for each assumption
    assumptionList = [i for i in assumptionTests.values()]
    assumptions = list(set(assumptionList))

    summaryTextList = []
    summarySentence = ' diagnostic tests passed for assumption - '

    for assumption in assumptions:
        testsFailed = violated.count(assumption)
        testsPerformed = assumptionList.count(assumption)
        testsPassed = testsPerformed - testsFailed

        sentence = str(testsPassed) + '/' + str(
                testsPerformed) + summarySentence + assumption
        if testsFailed > 0:
            print("\nFURTHER INSPECTION REQUIRED - CHECK PLOTS + DATA \n" +
                  sentence)
        summaryTextList.append(sentence)

# SAVE DIAGNOSTICS INFO AND SUMMARY OF DIAGNOSTIC TESTS
    # write out text file with summary of tests
    summaryFile = saveto + '\\' + step + '_testSummary.txt'
    with open(summaryFile, 'w') as f:
        for item in summaryTextList:
            f.write("%s\n" % item)

    csvName = saveto + '\\' + step + '_diagnostic_results.csv'
    # saves a csv with 28 rows and two columns (long but easily readable)
    pd.Series(diagnostics).to_csv(csvName)
    # Alternative code
    # saves a csv with 1 row and 28 columns (wide but not easily readable)
#    pd.DataFrame.from_dict(
#            diagnostics, orient='index').transpose().to_csv(csvName)
    # requires a roundabout way of creating the dataframe as regular method of
    # pd.DataFrame.from_dict(diagnostics) creates an empty df

    # save csv of pairwise correlations - only if there are multiple predictors
    if len(X.columns) > 1:
        pairwiseCorrName = saveto + '\\' + step + '_pairwise_correlations.csv'
        high_pairwise_corr.to_csv(pairwiseCorrName)

# MAKE AND SAVE PLOTS

# PLOT 1 - STUDENTISED RESIDUALS VS FITTED VALUES
# Used to inspect linearity and homoscedasticity
    # get values
    student_resid = influence_df['student_resid']
    fitted_vals = model.fittedvalues
    # plot with a LOWESS (Locally Weighted Scatterplot Smoothing) line
    # a relativelty straight LOWESS line indicates a linear model is reasonable
    residsVsFittedVals_plot = plt.figure()
    residsVsFittedVals_plot.axes[0] = sns.residplot(
            fitted_vals, student_resid, lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    residsVsFittedVals_plot.axes[0].set(ylim=(-3.5, 3.5))
    residsVsFittedVals_plot.axes[0].set_title('Residuals vs Fitted')
    residsVsFittedVals_plot.axes[0].set_xlabel('Fitted values')
    residsVsFittedVals_plot.axes[0].set_ylabel('Studentised Residuals')
    # name + save plot
    figName = saveto + '\\' + step + '_residualsVSfittedValuesPlot.png'
    residsVsFittedVals_plot.savefig(figName)
    plt.clf()

# PLOT 2 - NORMAL QQ PLOT OF RESIDUALS
# Used to inspect normality
    qq_fig = sm.qqplot(model.resid, fit=True, line='45')
    qq_fig.axes[0].set_title('Normal QQ Plot of Residuals')
    # name + save plot
    figName = saveto + '\\' + step + '_NormalQQPlot.png'
    qq_fig.savefig(figName)
    plt.clf()

# PLOT 3 - INFLUENCE PLOT WITH COOK'S DISTANCE
# Used to inspect influence
    # Outliers/Leverage/Influence - Influence plot w/ Cook's Distance & boxplot
    # https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.OLSInfluence.plot_influence.html#statsmodels.stats.outliers_influence.OLSInfluence.plot_influence
    fig_influence, ax_influence = plt.subplots(figsize=(12, 8))
    fig_influence = sm.graphics.influence_plot(model, ax=ax_influence,
                                               criterion="cooks")
    # name + save plot
    figName = saveto + '\\' + step + '_InfluencePlot_CooksD.png'
    fig_influence.savefig(figName)
    plt.clf()

# PLOT 4 - BOX PLOT OF STANDARDISED RESIDUALS
# Used to inspect outliers (residuals)
    outlier_fig = sns.boxplot(y=influence_df['standard_resid'])
    outlier_fig = sns.swarmplot(y=influence_df['standard_resid'], color="red")
    outlier_fig.axes.set(ylim=(-3.5, 3.5))
    outlier_fig.axes.set_title('Boxplot of Standardised Residuals')
    residBoxplot = outlier_fig.get_figure()  # get figure to save
    # name + save plot
    figName = saveto + '\\' + step + '_ResidualsBoxplot.png'
    residBoxplot.savefig(figName)
    plt.clf()

# PLOT 5 - PARTIAL REGRESSION PLOTS
# Used to inspect linearity
    # Partial regression plots
    fig_partRegress = plt.figure(figsize=(12, 8))
    fig_partRegress = sm.graphics.plot_partregress_grid(model,
                                                        fig=fig_partRegress)
    # name + save plot
    figName = saveto + '\\' + step + '_PartialRegressionPlots.png'
    fig_partRegress.savefig(figName)
    plt.clf()

# Return list of assumptions that require further inspection (i.e. failed
# at least 1 diagnostic test)
    assumptionsToCheck = list(set(violated))
    return assumptionsToCheck