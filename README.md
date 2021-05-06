# human_infections

This repository contains the R files necessary to run the analysis performed in the paper "Expression patterns of Plasmodium falciparum clonally variant genes at the onset of a blood infection in non-immune humans".

The analysis consists in the comparison, through gene expression microarrays, of the transcriptome of parasites obtained from infected volunteers and their parental parasite line maintained in culture.

This is the sessionInfo() output:

R version 3.6.3 (2020-02-29)
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Ubuntu 18.04.4 LTS

Matrix products: default
BLAS:   /usr/lib/x86_64-linux-gnu/openblas/libblas.so.3
LAPACK: /usr/lib/x86_64-linux-gnu/libopenblasp-r0.2.20.so

locale:
 [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C               LC_TIME=es_ES.UTF-8        LC_COLLATE=en_US.UTF-8     LC_MONETARY=es_ES.UTF-8   
 [6] LC_MESSAGES=en_US.UTF-8    LC_PAPER=es_ES.UTF-8       LC_NAME=C                  LC_ADDRESS=C               LC_TELEPHONE=C            
[11] LC_MEASUREMENT=es_ES.UTF-8 LC_IDENTIFICATION=C       

attached base packages:
[1] parallel  stats4    stats     graphics  grDevices utils     datasets  methods   base     

other attached packages:
 [1] sp_1.4-5             RColorBrewer_1.1-2   forcats_0.5.1        stringr_1.4.0        dplyr_1.0.5          purrr_0.3.4          readr_1.4.0         
 [8] tidyr_1.1.3          tibble_3.1.1         tidyverse_1.3.1      ggfortify_0.4.11     ggplot2_3.3.3        reshape2_1.4.4       AnnotationDbi_1.46.1
[15] IRanges_2.18.3       S4Vectors_0.22.1     Biobase_2.44.0       BiocGenerics_0.30.0 

loaded via a namespace (and not attached):
 [1] Rcpp_1.0.6       lattice_0.20-41  lubridate_1.7.10 assertthat_0.2.1 utf8_1.2.1       R6_2.5.0         cellranger_1.1.0 plyr_1.8.6      
 [9] backports_1.2.1  reprex_2.0.0     RSQLite_2.2.6    httr_1.4.2       pillar_1.6.0     rlang_0.4.10     readxl_1.3.1     rstudioapi_0.13 
[17] blob_1.2.1       bit_4.0.4        munsell_0.5.0    broom_0.7.6      compiler_3.6.3   modelr_0.1.8     pkgconfig_2.0.3  tidyselect_1.1.0
[25] gridExtra_2.3    fansi_0.4.2      crayon_1.4.1     dbplyr_2.1.1     withr_2.4.2      grid_3.6.3       jsonlite_1.7.2   gtable_0.3.0    
[33] lifecycle_1.0.0  DBI_1.1.1        magrittr_2.0.1   scales_1.1.1     cli_2.4.0        stringi_1.5.3    cachem_1.0.4     fs_1.5.0        
[41] xml2_1.3.2       ellipsis_0.3.1   generics_0.1.0   vctrs_0.3.7      tools_3.6.3      bit64_4.0.5      glue_1.4.2       hms_1.0.0       
[49] fastmap_1.1.0    colorspace_2.0-0 rvest_1.0.0      memoise_2.0.0    haven_2.4.0     
