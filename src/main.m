%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author    : Ranga Kulathunga
% Topic     : Digital Modulation Techniques
% Data      : May, 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear; clc;

%% Design - 1

%% (a) Non-ISI channel + symbol detection

% 1. Generate a random sequence of 16-QAM symbols
T = 1; % symbol time 
M = 10^4; % sequence length
qam_index = 16; % QAM index
k = log2(qam_index); % # of bits per symbol
bin_seq = randi([0 1],M*k,1); % generate binary sequence
sym_seq = bit2int(bin_seq,k); % convert binary sequence to integers
mod_seq = qammod(sym_seq,qam_index); % Gray-encoded

% 2. Generate transmitted signal using root-raised-cosine pulse
rolloff = 0.35;
filtlen = 40;
sps = 10;
rrcFilter = rcosdesign(rolloff,filtlen,sps);
s_t = upfirdn(mod_seq,rrcFilter,sps,1);

snr_array = 0:2:26;
ber_array = zeros(1,length(snr_array));
for i = 1:length(snr_array)
    % 3. Pass the signal through the channel and the matched filter
    snr = snr_array(i);
    r_t = awgn(s_t,snr,'measured');
    y = upfirdn(r_t,rrcFilter,1,sps);
    y = y(filtlen + 1:end - filtlen);
    
    % 4. Generate decision variables
    sym_seq_hat = qamdemod(y,qam_index);
    bin_seq_hat = int2bit(sym_seq_hat,k);

    % 5. BER calculations
    [numErrors,ber] = biterr(bin_seq,bin_seq_hat);
    % ber = sum(abs(sym_seq_hat-sym_seq))/(M*k);
    ber_array(i) = ber;
end

figure
semilogy(snr_array, ber_array)
xlim([0 26])
xlabel("SNR (dB)")
ylabel("Average BER")

%% (a) ISI channel + symbol detection

% 1. Generate a random sequence of 16-QAM symbols
T = 1; % symbol time 
M = 10^4; % sequence length
qam_index = 16; % QAM index
k = log2(qam_index); % # of bits per symbol
bin_seq = randi([0 1],M*k,1); % generate binary sequence
sym_seq = bit2int(bin_seq,k); % convert binary sequence to integers
mod_seq = qammod(sym_seq,qam_index); % Gray-encoded

% 2. Generate transmitted signal using root-raised-cosine pulse
rolloff = 0.35;
filtlen = 60;
sps = 10;
rrcFilter = rcosdesign(rolloff,filtlen,sps);
s_t = upfirdn(mod_seq,rrcFilter,sps,1);

beta_array = [0, 0.01, 0.1, 0.2];
snr_array = 0:2:26;
ber_mat = zeros(length(beta_array), length(snr_array));
figure
for i = 1:length(beta_array)
    % 3. Pass the signal through the channel and the matched filter
    beta = beta_array(i);
    h_t = filter([1 beta], 1, s_t);
    for j = 1:length(snr_array)
        snr = snr_array(j);
        r_t = awgn(h_t,snr,'measured');
        y = upfirdn(r_t,rrcFilter,1,sps);
        y = y(filtlen + 1:end - filtlen);
        
        % 4. Generate decision variables
        sym_seq_hat = qamdemod(y,qam_index);
        bin_seq_hat = int2bit(sym_seq_hat,k);
    
        % 5. BER calculations
        [numErrors,ber] = biterr(bin_seq,bin_seq_hat);
        % ber = sum(abs(sym_seq_hat-sym_seq))/(M*k);
        ber_mat(i,j) = ber;
    end
    semilogy(snr_array, ber_mat(i,:))
    hold on
end
xlim([0 26])
xlabel("SNR (dB)")
ylabel("Average BER")
legend("\beta=0", "\beta=0.01", "\beta=0.1", "\beta=0.2")