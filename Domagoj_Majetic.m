%Programski zadatak 
%Domagoj Majetic, 0035215314

clc
clear all

%Unos problema 
problem_input = input('Unesite "x" za XOR problem ili bilo koji drugi unos za proizvoljan problem iz datoteka D.txt i Z.txt: ', 's');
if problem_input == 'x'
    D = [0; 1; 1; 0];
    Z = [0 0 1; 1 0 1; 0 1 1; 1 1 1]; 
else
    D = load('D.txt');
    Z = load('Z.txt');
    ZBias = ones(size(Z,1), 1);
    Z = [Z ZBias];
end

%Unos pocetnih tezina
tezine_input = input('Unesite "t" za tezine s predavanja ili bilo koji drugi unos za proizvoljne tezine: ', 's');
if tezine_input == 't'
   V = [0.2 0.9 -0.3; -0.9 0.8 0.2];
   W = [-0.2 -0.9 0.7];
   br_skrivenih = size(V,1);
else 
    random_tezine_input = input('Unesite "r" za nasumicne tezine ili bilo koji drugi unos za tezine iz datoteka V.txt i W.txt: ', 's');
    if random_tezine_input == 'r'
        br_ulaz = size(Z,2);
        br_skrivenih = input ('Unesite zeljeni broj neurona skrivenog sloja (npr. "5"): ');
        br_izlaz = size(D,2);
        V = -1+rand(br_skrivenih, br_ulaz)*2;
        W = -1+rand(br_izlaz, br_skrivenih + 1)*2;
    else
        V = load('V.txt');
        W = load('W.txt');
        br_skrivenih = size(V,1);
    end
end

%Odabir aktivacijske funkcije i zeljenih vrijednosti
aktfun = input('Unesite "1" ili "2" za odabir aktivacijske funkcije: ');
while aktfun ~= 1 && aktfun ~= 2
    aktfun = input('Pogresan unos! Ponovno unesite "1" ili "2" za odabir aktivacijske funkcije: ');
end
eta = input('Unesite vrijednost zeljenog koeficijenta ucenja (npr. "0.1"): ');
br_koraka = input('Unesite zeljeni broj koraka ucenja (npr. "1000"): ');
desired_NRMS = input('Unesite vrijednost zeljenog NRMS-a (npr. "0.05"): ');
alpha = input('Unesite vrijednost za zeljeni momentum (npr. "0.5"): ');

O = zeros(size(D));
MS = 0;
sigma = std(D,1);
NRMS = zeros(br_koraka, size(Z,2));
V_old = V;
W_old = W;

%Odziv 0-tog koraka
for i = 1 : size(Z,1)
    netH = V*Z(i,:)';
    if aktfun == 1
        y = [-1 + 2./(1 + exp(-netH)); 1];
    elseif aktfun == 2
        y = [1./(1 + exp(-netH)); 1];
    end
    O (i,:) = W*y;
end

%Faze ucenja 
for n = 1 : br_koraka
    for i = 1 : size(Z,1)
        netH = V*Z(i,:)';
        if aktfun == 1
            y = -1 + 2./(1 + exp(-netH));
            y_ = y;
            y = [-1 + 2./(1 + exp(-netH)); 1];
        elseif aktfun == 2
            y = 1./(1 + exp(-netH));
            y_ = y;
            y = [1./(1 + exp(-netH)); 1];
        end
        O = W*y;
        
        nablaEW = -(D(i,:)' - O)*y';
        W_new = W - eta*nablaEW + alpha*(W - W_old);
        DEy = - (D(i,:) - O')*W(:, 1 : br_skrivenih);
        if aktfun == 1
            DynetH = 0.5*(1-y_(1 : br_skrivenih).^2);
        elseif aktfun == 2
            DynetH =(1./(exp(netH)+1)).*y_(1 : br_skrivenih);
        end
        nablaEV = DEy'.*DynetH*Z(i,:);
        V_new = V - eta*nablaEV + alpha*(V - V_old);
        
        V_old = V;
        W_old = W;
        
        V = V_new;
        W = W_new;
    end
    netH_ = V*Z';
    if aktfun == 1
        y_nrms = [-1 + 2./(1 + exp(-netH_)); ones(1, size(netH_,2))];
    elseif aktfun == 2
        y_nrms = [1./(1 + exp(-netH_)); ones(1, size(netH_,2))];
    end
    O = W*y_nrms;
    MS = sum((D - O').^2)/size(Z,1);
    RMS = sqrt(MS);
    NRMS(n,:) = RMS/sigma;
    if desired_NRMS > max(NRMS(n, :))
        NRMS = NRMS(1:n, :);
        break
    end
    MS = 0;
end

O_ = zeros(size(D));
for i = 1 : size(Z,1)
    netH = V*Z(i,:)';
    if aktfun == 1
        y = [-1 + 2./(1 + exp(-netH)); 1];
    elseif aktfun == 2
        y = [1./(1 + exp(-netH)); 1];
    end
    O_(i,:) = W*y;
end

%Graf
plot(1:n, NRMS, 'Color', [1 0 0], 'LineWidth', 2)
grid on
xlabel('Korak ucenja'),ylabel('NRMS')

%Ispis rezultata
disp('Broj koraka ucenja: ')
disp(n)
disp('Zeljeni izlaz: ')
disp(D)
disp('Dobiveni izlaz: ')
disp(O)
disp('Iznos NRMS-a: ')
disp(NRMS(n))
