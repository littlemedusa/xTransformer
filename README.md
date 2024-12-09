$$
\begin{tabular}{|c|c|cc|cc|cc|cc|cc|}
\hline 
 \multicolumn{2}{|c|}{Methods} & \multicolumn{2}{c|}{t\_combined\_t} & \multicolumn{2}{c|}{PatchTST\_21} & \multicolumn{2}{c|}{PatchTST} & \multicolumn{2}{c|}{t\_combined\_n} & \multicolumn{2}{c|}{iTransformer} \\
\hline 
 \multicolumn{2}{|c|}{description} & \multicolumn{2}{c|}{(BN), [T, 1-dk]} & \multicolumn{2}{c|}{(BN), [T, 1-dk]} & \multicolumn{2}{c|}{(BN), [T, T-dk]} & \multicolumn{2}{c|}{(B), [N, T-dk]} & \multicolumn{2}{c|}{(B), [N, T-dk]} \\
\hline 
 \multicolumn{2}{|c|}{Metric} & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE \\
\hline 
 \multirow{2}{*}{ETTh1} & 96 & 0.375 & 0.401 & 0.376 & 0.395 & 0.414 & 0.419  & 0.393 & 0.413 & 0.386  & 0.405  \\
  & 336 & 0.451 & 0.445 & 0.459 & 0.442 & 0.501 & 0.466  & 0.476 & 0.456 & 0.487  & 0.458  \\
\hline 
 \multirow{2}{*}{ETTh2} & 96 & 0.288 & 0.337 & 0.287 & 0.341 & 0.302  & 0.348  & 0.300 & 0.350 & 0.297  & 0.349  \\
  & 336 & 0.416 & 0.430 & 0.416 & 0.428 & 0.426  & 0.433  & 0.427 & 0.436 & 0.428  & 0.432  \\
\hline 
 \multirow{2}{*}{ETTm1} & 96 & 0.316 & 0.356 & 0.328 & 0.360 & 0.329 & 0.367 & 0.338 & 0.372 & 0.334 & 0.368 \\
  & 336 & 0.392 & 0.401 & 0.406 & 0.404 & 0.399 & 0.410 & 0.417 & 0.415 & 0.426 & 0.420 \\
\hline 
 \multirow{2}{*}{ETTm2} & 96 & 0.181 & 0.262 & 0.178 & 0.263 & 0.175  & 0.259  & 0.188 & 0.274 & 0.180  & 0.264  \\
  & 336 & 0.311 & 0.351 & 0.306 & 0.347 & 0.305  & 0.343  & 0.315 & 0.352 & 0.311  & 0.348  \\
\hline 
 \multirow{2}{*}{Exchange} & 96 & 0.081 & 0.197 & 0.082 & 0.199 & 0.088  & 0.205  & 0.086 & 0.205 & 0.086  & 0.206  \\
  & 336 & 0.318 & 0.408 & 0.317 & 0.406 & 0.319  & 0.408  & 0.318 & 0.409 & 0.331  & 0.417  \\
\hline 
 \multirow{2}{*}{Weather} & 96 & 0.168 & 0.215 & 0.190 & 0.229 & 0.177  & 0.218  & 0.180 & 0.219 & 0.174  & 0.214  \\
  & 336 & 0.271 & 0.296 & 0.285 & 0.301 & 0.278  & 0.297  & 0.280 & 0.299 & 0.278  & 0.296  \\
\hline 
 \multirow{2}{*}{ECL} & 96 & 0.173 & 0.258 & 0.198 & 0.282 & 0.195 & 0.285 & 0.151 & 0.243 & 0.148 & 0.240 \\
  & 336 & 0.204 & 0.289 & 0.212 & 0.299 & 0.215 & 0.305 & 0.178 & 0.272 & 0.178 & 0.269 \\
\hline 
 \multirow{2}{*}{Solar} & 96 & 0.205 & 0.251 & 0.274 & 0.332 & 0.234 & 0.286 & 0.225 & 0.242 & 0.203 & 0.237 \\
  & 336 & 0.259 & 0.288 & 0.315 & 0.329 & 0.290 & 0.315 & 0.253 & 0.277 & 0.248 & 0.273 \\
\hline 
 \multirow{2}{*}{Traffic} & 96 & 0.422 & 0.282 & 0.550 & 0.379 & 0.544 & 0.359 & 0.471 & 0.344 & 0.395 & 0.268 \\
  & 336 & 0.424 & 0.278 & 0.491 & 0.326 & 0.551 & 0.358 & 0.500 & 0.361 & 0.433 & 0.283 \\
 \hline
\end{tabular}
$$
