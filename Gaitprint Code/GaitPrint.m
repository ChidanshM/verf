
% TITLE: GaitPrint.m
% DATE: September 1, 2022
% AUTHOR: Tyler M. Wiles, MS
% EMAIL: tylerwiles@unomaha.edu

% DESCRIPTION:
% This code is intended to be a single script designed to run a complete
% analysis on data collected for the Gaitprints as Predictors for Disease
% and Disability for Rehabilitation Engineering project. The data that will
% be used is 4 minutes of overground, self selected pace, walking using a
% full body Noraxon Ultium IMU configuration. Gaitprint.m will loop through
% each trial, located in a single folder, apply all functions designed for
% this specific project, and export everything into .mat files per
% participant.

% NOTE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code will not run unless it is routed to a specific folder format.
% Each individual trial must be its own file within a single folder with
% the following naming structure "S001_G01_D01_B01_T01". Basically, the
% Raw_Data cells from each subject's .mat file must be written to their own
% file to be used with this code.

% Copyright 2022, Tyler M. Wiles

% Redistribution and use of this script, with or without
% modification, is permitted provided this copyright notice,
% the original authors name and the following disclaimer remains.

% DISCLAIMER: It is the user's responsibility to check the code is returning
% the appropriate results before reporting any scientific findings.
% The author will not accept any responsibility for incorrect results
% directly relating to user error.

%%

clear all; close all; clc;

% This line will allow us to use the functions in the FUNCTION folder
addpath('FUNCTION')

my_directory = 'Your filepath here'; % Hard code our directory

% This is the folder where our results will go
output_directory = append(my_directory, '\GAITPRINT OUTPUT');

% Identify all files beginning with S
my_files = dir(fullfile(my_directory,'CLEAN DATA', 'S*'));

% This is where our outputs will go
gaitprint_results = [];
temp_results = []; % Temporary storage for concatenating new values

sampling_rate = 200; % Sampling rate used to capture data

% Columns we will use to name files
Gaitprint_Noraxon_Names_Clean = readcell('Gaitprint_Noraxon_Names_Clean.xlsx');
Gaitprint_Noraxon_Names_Clean = Gaitprint_Noraxon_Names_Clean(2:end); % Removing time
headers = readcell("Gaitprint_Noraxon_Names_Clean.csv");

% Options for spaiotemporal calculation
plot_option = 0; save_option = 1;

% Loop setup - Change according to specific files used
iteration_start = 1; iteration_end = length(my_files);
count = 0; % Necessary for initial loop condition

for i = iteration_start:iteration_end
    % Getting file names that will be useful for the rest of the code
    base_filename = my_files(i).name;
    full_filename = fullfile(my_directory, 'CLEAN DATA', base_filename);
    
    % Display file being used
    disp('Analyzing');
    disp(base_filename);
    
    % Read in data
    dat = readmatrix(full_filename);
    clean_dat = array2table(dat);
    clean_dat.Properties.VariableNames = headers;
    
    % Extracting filenames with conditions
    filename_split = strsplit(base_filename,'.');
    id_condition = strsplit(filename_split{1},'_');
    
    % Extracting filenames to create figures
    png_filename = filename_split{1};
    png = append(png_filename, '.png');
    
    count = count + i; % Necessary for initial loop condition
    
    %% Spatiotemporal Analysis
    
    % Calculate Spatiotemporal gait characteristics
    [spatiotemporal_variables] = GaitPrint_Spatiotemporal_Calculation(base_filename,...
        dat, sampling_rate, png_filename, png, plot_option, save_option);
    
    % Add the data into a table in preparation for exporting
    temp_results{1,1} = id_condition(1);
    temp_results{1,2} = id_condition(2);
    temp_results{1,3} = id_condition(3);
    temp_results{1,4} = id_condition(4);
    temp_results{1,5} = id_condition(5);
    temp_results{1,6} = clean_dat;
    temp_results{1,7} = spatiotemporal_variables;
    
    %% Exporting Sequence
    
    % If the subject ID of temp_results is THE SAME as gaitprint_results then
    % concatenate by row. If the subject ID of temp_results is NOT THE SAME as
    % gaitprint_results then save gaitprint_results, clear gaitprint results,
    % move temp_results to gaitprint_results, clear temp_results for next
    % iteration.
    
    % If gaitprint_results is empty, fill it with temp_results. I am using
    % iteration_start so this always occur on the first trial regardless of
    % the individual trial number.
    if count == iteration_start
        
        gaitprint_results = cat(1, gaitprint_results, temp_results);
        
    else
        
        % If the subject ID of temp_results is THE SAME as gaitprint_results AND it
        % is not the last iteration, then concatenate to gaitprint_results, then
        % clear for next iteration
        if isequal(temp_results(end,1),gaitprint_results(1)) & i ~= iteration_end
            
            gaitprint_results = cat(1, gaitprint_results, temp_results);
            temp_results = [];
            
            % If the subject ID of temp_results is THE SAME as gaitprint_results AND it
            % is the last iteration, then concatenate and save
        else if isequal(temp_results(end,1),gaitprint_results(1)) & i == iteration_end
                
                gaitprint_results = cat(1, gaitprint_results, temp_results);
                
                disp('Saving......')
                s = gaitprint_results{1};
                % Changing dataset to a table
                subject_data = cell2table(gaitprint_results);
                
                % Creating variable names for each column
                trial_characteristics = {'ID','Group','Day','Block','Trial','Raw_Data'};
                distance_spatiotemporal_names = {'Spatiotemporal_Variables'};
                gaitprint_results_variable_names = [trial_characteristics, distance_spatiotemporal_names];
                subject_data.Properties.VariableNames = gaitprint_results_variable_names;
                
                % Save
                save([output_directory, '\', s{1}, '_data'], 'subject_data', '-v7.3');
                
                % If the subject ID of temp_results is NOT THE SAME as gaitprint_results,
                % then save gaitprint_results, clear gaitprint results, move temp_results
                % to gaitprint_results, clear temp_results for next iteration.
            else
                
                disp('Saving......')
                s = gaitprint_results{1};
                % Changing dataset to a table
                subject_data = cell2table(gaitprint_results);
                
                % Creating variable names for each column
                trial_characteristics = {'ID','Group','Day','Block','Trial','Raw_Data'};
                distance_spatiotemporal_names = {'Spatiotemporal_Variables'};
                gaitprint_results_variable_names = [trial_characteristics, distance_spatiotemporal_names];
                subject_data.Properties.VariableNames = gaitprint_results_variable_names;
                
                % Save
                save([output_directory, '\', s{1}, '_data'], 'subject_data', '-v7.3');
                
                gaitprint_results = [];
                gaitprint_results = cat(1, gaitprint_results, temp_results);
                temp_results = [];
                
            end
            
        end
        
    end
    
end

