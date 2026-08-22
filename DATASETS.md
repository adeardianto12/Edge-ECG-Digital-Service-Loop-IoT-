# Dataset Access and Attribution

The repository does not redistribute ECG waveforms or annotations. Download
each database from its official PhysioNet page and comply with the applicable
terms of use and citation requirements.

| Database | Role in this project | Official page |
|---|---|---|
| MIT-BIH Arrhythmia | Development | https://physionet.org/content/mitdb/ |
| INCART | Development | https://physionet.org/content/incartdb/ |
| Long-Term AF Database (LTDDB) | Development supplement | https://physionet.org/content/ltdb/ |
| Sudden Cardiac Death Holter Database (SVDB) | Frozen external evaluation | https://physionet.org/content/svdb/ |
| Normal Sinus Rhythm Database (NSRDB) | Normal-rhythm false-alarm evaluation | https://physionet.org/content/nsrdb/ |

The local audit records database roles, record groups, selected leads,
resampling, and hashes. See `AGENTS.md` and the published preprocessing
contract summary. SVDB and NSRDB must not be used for model selection.
