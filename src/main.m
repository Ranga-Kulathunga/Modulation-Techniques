%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author    : Ranga Kulathunga
% Topic     : Digital Modulation Techniques
% Data      : May, 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clear; clc;

%% Design - 1

%% (a) Non-ISI channel + symbol detection

disp("------------------ Non-ISI Channel + Symbol Detection ------------------")
% 1. Generate a random sequence of 16-QAM symbols
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
    ber_array(i) = ber;
    disp(sprintf("BER=%f for SNR=%d dB",ber,snr));
end

figure
semilogy(snr_array, ber_array)
xlim([0 26])
xlabel("SNR (dB)")
ylabel("Average BER")

%% (b) ISI channel + symbol detection

disp("------------------ ISI Channel + Symbol Detection ------------------")
% 1. Generate a random sequence of 16-QAM symbols
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
        ber_mat(i,j) = ber;
        disp(sprintf("BER=%f for SNR=%d dB and Beta=%f",ber,snr,beta));
    end
    semilogy(snr_array, ber_mat(i,:))
    hold on
end
xlim([0 26])
xlabel("SNR (dB)")
ylabel("Average BER")
legend("\beta=0", "\beta=0.01", "\beta=0.1", "\beta=0.2")

%% (c) ISI channel + sequence detection

disp("------------------ ISI Channel + Sequence Detection ------------------")
% 1. Generate a random sequence of 16-QAM symbols
M = 10^4; % sequence length
qam_index = 16; % QAM index
k = log2(qam_index); % # of bits per symbol
bin_seq = randi([0 1],M*k,1); % generate binary sequence
sym_seq = bit2int(bin_seq,k); % convert binary sequence to integers
mem_symbol = -3+1j*3;
mod_seq = [qammod(sym_seq,qam_index); mem_symbol]; % Gray-encoded
L = log2(qam_index);
c = 1;
const_mat = repelem((-(L-1)*c:2*c:(L-1)*c),L,1)+1j*repelem(((L-1)*c:-2*c:-(L-1)*c)',1,L);

% 2. Generate transmitted signal using root-raised-cosine pulse
rolloff = 0.35;
filtlen = 20;
sps = 10;
rrcFilter = rcosdesign(rolloff,filtlen,sps);
s_t = upfirdn(mod_seq,rrcFilter,sps,1);

beta_array = [0.01, 0.1, 0.2];
snr_array = 0:2:26;
metric_mat = zeros(qam_index,qam_index,M+1);
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
    
        % 4. Run Viterbi algorithm
        S_k = zeros(qam_index,qam_index);
        for n = 1:qam_index
            for m = 1:qam_index
                S_k(n,m) = const_mat(m)+beta*const_mat(n);
            end
        end
        out_mat = zeros(1,1,M-1);
        out_mat(1,1,:) = y(2:end-1);
        metric_mat(:,:,2:end-1) = abs(out_mat-S_k).^2;
        metric_mat(1,:,1) = abs(S_k(1,:)-y(1)).^2;
        metric_mat(:,1,end) = abs(S_k(:,1)-y(end)).^2;
    
        acc_mat = zeros(qam_index,M+1);
        pred_seq_mat = zeros(qam_index,M+1);
        for n = 1:M+1
            if(n == 1)
                acc_mat(:,1) = reshape(metric_mat(1,:,1),qam_index,1);
                pred_seq_mat(:,1) = const_mat(1);
            elseif(n == M+1)
                metric = acc_mat(:,n-1)+metric_mat(:,1,n);
                min_metric = min(metric);
                sel = find(metric == min_metric);
                acc_mat(1,n) = min_metric;
                pred_seq_mat(1,n) = const_mat(sel);
            else
                for m = 1:qam_index
                    metric = acc_mat(:,n-1)+metric_mat(:,m,n);
                    min_metric = min(metric);
                    sel = find(metric == min_metric);
                    acc_mat(m,n) = min_metric;
                    pred_seq_mat(m,n) = const_mat(sel);
                end
            end
        end
        y_hat = zeros(M+1,1);
        y_hat(M+1) = mem_symbol;
        for n = M+1:-1:2
            if(n == M+1)
                y_hat(n-1) = pred_seq_mat(1,M+1);
            else
                sym_arr = pred_seq_mat(:,n);
                sym_index = find(const_mat == y_hat(n));
                y_hat(n-1) = sym_arr(sym_index);
            end
        end
    
        % 5. Generate decision variables
        sym_seq_hat = qamdemod(y_hat,qam_index);
        bin_seq_hat = int2bit(sym_seq_hat,k);
    
        % 6. BER calculations
        [numErrors,ber] = biterr([bin_seq;zeros(k,1)],bin_seq_hat);
        ber_mat(i,j) = ber;
        disp(sprintf("BER=%f for SNR=%d dB and Beta=%f",ber,snr,beta));
    end
    semilogy(snr_array, ber_mat(i,:))
    hold on
end
xlim([0 26])
xlabel("SNR (dB)")
ylabel("Average BER")
legend("\beta=0.01", "\beta=0.1", "\beta=0.2")

%% Design - 2

% Orthogonal Frequency Division Multiplexing (OFDM)
disp("------------------ OFDM ------------------")

%% (b) Symbol Error Probability

disp("------------------ SEP ------------------")
M = 10^4; % sequence length
qam_index = 16; % QAM index
k = log2(qam_index); % # of bits per symbol
N = 64; % FFT size or total number of subcarriers
N_cp = 16;
L=10; % Number of taps for the frequency selective channel model
snr_array_dB = 0:2:26; % SNR per bit in dB scale
sym_err_arr_sim = zeros(1, length(snr_array_dB));
for i = 1:length(snr_array_dB)
    snr = snr_array_dB(i);
    sym_error = 0;
    for j = 1:M
        % 1. Generate transmit signal
        bin_seq = randi([0 1],N*k,1); % generate binary sequence
        sym_seq = bit2int(bin_seq,k); % convert binary sequence to integers
        mod_seq = qammod(sym_seq,qam_index); % Gray-encoded
        mod_seq_ifft = ifft(mod_seq,N);
        s_t = [mod_seq_ifft(end-N_cp+1:end); mod_seq_ifft];
        
        % 2. Propagate through Rayleigh and noisy channel
        h = 1/sqrt(2)*(randn(L,1)+1i*randn(L,1));
        H = fft(h,N);
        h_s = conv(h,s_t);
        P = 1*sum(abs(h_s).^2)/length(h_s);
        gamma = 10^(0.1*snr)*k*N/(N+N_cp);
        N0 = P/gamma;
        n = sqrt(N0/2)*(randn(size(h_s))+1i*randn(size(h_s)));
        r_t = h_s + n;

        % 3. Receiver signal
        y = r_t(N_cp+1:N+N_cp);
        y_fft = fft(y,N);
        V = y_fft./H; % Equalization
        
        % 4. Generate decision variables
        sym_seq_hat = qamdemod(V,qam_index);
    
        % 5. Symbol Error calculations
        sym_error = sym_error + sum(sym_seq_hat ~=sym_seq)/N;
    end
    sym_err_arr_sim(i) = sym_error/M;
    disp(sprintf("BER=%f for SNR=%d dB",sym_err_arr_sim(i),snr));
end
snr_array = k*10.^(0.1*snr_array_dB); % SNR per symbol in linear scale
sym_err_arr_an_1 = 2*(1-1/sqrt(qam_index))*(1-sqrt(1.5*snr_array./(qam_index-1+1.5*snr_array)));
sym_err_arr_an_2 = (1-1/sqrt(qam_index))^2*(1-4/pi*sqrt(1.5*snr_array./(qam_index-1+1.5*snr_array)).*atan(sqrt((qam_index-1+1.5*snr_array)./(1.5*snr_array))));
sym_err_arr_an = sym_err_arr_an_1 - sym_err_arr_an_2;

figure
semilogy(snr_array_dB, sym_err_arr_sim)
hold on
semilogy(snr_array_dB, sym_err_arr_an)
xlim([0 26])
xlabel("SNR (dB)")
ylabel("Average Symbol Error Rate")
legend("Simulation","Analytical")
