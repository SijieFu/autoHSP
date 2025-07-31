# Experiment Records from the Lab
** This folder contains the experiment records for each test resion in the lab.**

- The summary of the records is stored in the `summary.json` file.
  - thread keys: the value is the task id of the tasks that this thread in running;
  - task id keys: the value is the task information, including:
    - `task`: the task name, should be one of `prep`, `image`, `pause`, `EOE`;
    - `task_id`: the task id;
    - `samples`: a list of dictionaries, each with the following keys:
      - `label`: the label of the sample;
      - `resin`: the resin;
      - `ramount`: the amount of resin to add;
      - `solvent`: the list of pure solvents;
      - `samount`: the list of solvent amounts to add.

- For each resin, its log file has a name of `{resin}.csv`

- If a resin has prior information about its miscibility with certain solvents, the information should be kept in file `{resin}_prior.csv` with the following columns
  - `Solvent`: optional, a column to store the solvent code name or original name;
    > In practice, HSP values (dD, dP, dH) are used to represent a solvent, so you are not required to have this column.
  - `dD`: HSP value for dispersion;
  - `dP`: HSP value for polarity;
  - `dH`: HSP value for hydrogen bonding;
  - `Result`: "Y" for yes (solvent is miscible with resin), "N" for no (solvent is NOT miscible with resin).
  - You can add more columns if needed for your own analysis.

- For the resin-specific experimentation log file `{resin}.csv`, it should have the following columns
  - `Label`: the label for the mixture to be preped in the lab;
  - `Resin`: code name of the resin;
  - `ResinAmount`: amount of resin used in the experiment, the default should be 5 (milliliter or gram, depending on the resin);
  - `Solvent`: colon-seperated (:) code name(s) of the solvent(s) used in the experiment, for example, "R1", "R2", and "R1,R2";
  - `SolventAmount`: colon-seperated (:) amount(s) of the solvent(s) used in the experiment, for example, "5" and "1,4" - should match the number of solvents used;
  - `dD`: HSP value for dispersion;
  - `dP`: HSP value for polarity;
  - `dH`: HSP value for hydrogen bonding;
  - `Initiator`: how is this experiment initiated, for example, "init" for the first round of exploration generation, "explore" for generation with some known information, "exploit" for generation with more known information to generate solvents close to the decision boundary.
  - `Result`: "Y" for yes (solvent is miscible with resin), "N" for no (solvent is NOT miscible with resin).
  - `ResultRevised`: if `result` needs or is already revised by human.
    - `0`: the corresponding result in `results` does not need revision and is not revised
    - `1`: the corresponding result in `results` is already revised
    - `-1`: the corresponding result in `results` needs revision (pending revision)
  - `LabId`: the lab-specific ID for the vessel that holds the mixture of resin and solvent;
  - `PrepTaskId`: the task ID for the prepSamples experiment in the `summary.json` file;
  - `PrepByThread`: the thread name that initiates the prepSamples experiment;
  - `PrepStart`: the start time of the prepSamples experiment in the lab;
  - `PrepEnd`: the end time of the prepSamples experiment in the lab;
  - `PrepProtocolId`: the prepSamples protocol ID in the lab;
  - `ImageTaskId`: the task ID for the imageSamples experiment in the `summary.json` file;
  - `ImageByThread`: the thread name that initiates the imageSamples experiment;
  - `ImageStart`: the start time of the imageSamples experiment in the lab;
  - `ImageEnd`: the end time of the imageSamples experiment in the lab;
  - `ImageProtocolId`: the imageSamples protocol ID in the lab.