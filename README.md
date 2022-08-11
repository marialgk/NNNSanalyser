# NNNS Analyser

**Version:** 1.0
**Author:** Maria Laura Gabriel Kuniyoshi

Compute the summary scores and draw graphs from raw data of the NeoNatal Neurobehavioral Scale (NNNS).

## Usage:

    python ./AnalyseNNNS.py file [-h] [--output_filename OUTPUT_FILENAME] [--file_type {xlsx,csv}] [--boxplot BOOL]

## Input:
Currently, NNNS Analyser support two input types:
- xlsx (latest Microsoft Excel file), according to the model available at this repository.
    - Please do not delete any row nor columns A, B and C.
    - Currently, the template is only available in Portuguese, English version may be added in a future version.
- csv (comma separated values), according to the sample available at this repository.
    - The first column must be the variable number ID.

## Output:
- TSV table with summary statistics of each summary score.
- Boxplots for each summary score, saved individually as PNG images.

Currently, NNNS Analyse does not offer options for customizing the Boxplots. Feel free to adapt the code, as long as you cite my work, if substantial parts are published elsewhere.

## Options:

- **File:** Required. Name of the file with the NNNS raw data. Must be one of the formats described above.
- **--output_filename:** Optional. String to be written as part of the output file name. Default is None. 
- **--file_type:** Optional. File type of the input table. Currently accepted options are csv and xlsx. Default is xlsx.
- **--boxplot:** Optional. Boolean. If True, produces and saves boxplots. Default is True.
 - **-h, --help:** Show help message.

## Further reading

Original NNNS publication:

Lester, B. M., Tronick, E. Z., & Brazelton, T. B. (2004). [The Neonatal Intensive Care Unit Network Neurobehavioral Scale procedures](https://doi.org/10.1542/peds.113.S2.641). Pediatrics, 113(3 Pt 2), 641–667.

Lester, B. M., Tronick, E. Z., & Brazelton, T. B. (2004). [Appendix 3: Summary Score Calculations]( https://doi.org/10.1542/peds.113.S2.695). Pediatrics, 113 (pt 2), 695–699. 
