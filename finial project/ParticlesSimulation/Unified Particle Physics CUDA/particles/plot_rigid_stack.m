clear;
y0 = importdata('0.txt');
y1 = importdata('1.txt');
y2 = importdata('2.txt');
y3 = importdata('3.txt');
y4 = importdata('4.txt');
y5 = importdata('5.txt');
t = 1:300
% y0 = y0(t);
% y1 = y1(t);
% y2 = y2(t);
% y3 = y3(t);
% y4 = y4(t);
% y5 = y5(t);
hold on
xlabel('frame')
ylabel('height')
plot(t,y0,'r-')
legend('k=0','k=1','k=2','k=3','k=4','k=5');
plot(t,y1,'b+')
plot(t,y2,'kv')
plot(t,y3,'ys')
plot(t,y4,'mx')
plot(t,y5,'g^')
