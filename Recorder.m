%% Gear 1
R = 20; N = 32; T = [-2,-1.3,-0.7, 0.7,1.3,2]; Z = [-1,-1,1,1,-1,-1];
L = R*sin(pi/N); T = T/2*L; theta = pi/2 - asin(T/R); 
X = R*cos(theta); Y = R*sin(theta); Path = [X;Y];  
M = [cos(2*pi/N), sin(2*pi/N); -sin(2*pi/N), cos(2*pi/N)];
for n = 1:N-1
    Path = M*Path; X = [X, Path(1,:)]; Y = [Y, Path(2,:)];
end
Z = repmat(Z,1,N); 
X = [0;0.2;0.5;0.5;1;1;0.2;0]*X; Y = [0;0.2;0.5;0.5;1;1;0.2;0]*Y; 
Z = [3*ones(3,numel(Z));7.8+0.5*Z;2+Z;zeros(3,numel(Z))]+60;
figure('color', 'w'); 
[gear1s, gear1l] = MakeGear(X, Y, Z, 4:5, [1,1,1], 0.5);

%% Gear 2
R = 10; N = 16; T = [-2,-1.3,-0.7, 0.7,1.3,2]; Y = [-1,-1,1,1,-1,-1]+1;
L = R*sin(pi/N); T = T/2*L; theta = pi/2 - asin(T/R); 
X = R*cos(theta); Y = Y + R*sin(theta);  Path = [X;Y]; 
M = [cos(2*pi/N), sin(2*pi/N); -sin(2*pi/N), cos(2*pi/N)];
for n = 1:N-1
    Path = M*Path; X = [X, Path(1,:)]; Y = [Y, Path(2,:)];
end
Z = [10;10;0;0]*ones(size(X))-20; 
X = 73.1+[0;0.5;1;0]*X; Y = [0;0.5;1;0]*Y; 
[gear2s, gear2l] = MakeGear(Y, -Z, X, 2:3, [1,1,1], 0.5);
RotateGear(gear2s, gear2l, pi/N, 2:3);

%% Axle
[Xc, Yc, Zc] = cylinder([0,4,4,20,20,15,15,20,20,4,4,0], 50); 
Zc(1:2,:) = 0; Zc(3:4,:) = 8; Zc(5:6,:) = 10;  Zc(7:8,:) = 55;
Zc(9:10,:) = 57; Zc(11:12,:) = 70; 
MakeGear(Xc, Yc, Zc, [],[1,1,1], 0.5);
MakeGear(Xc-15*pi, Yc, Zc, [],[1,1,1], 0.5);


%% Graph Sheet
t = (-50:150)*2*pi/200;
x = [15.1*cos(t)-15*pi, 15.1*cos(t)];
y = [15.1*sin(t), 15.1*sin(t)];
x_ink = [15.3*cos(t)-15*pi, 15.3*cos(t)];
y_ink = [15.3*sin(t), 15.3*sin(t)];
L = cumsum([0,sqrt(diff(x).^2 + diff(y).^2)])/15.1;
chartangle = (0:100)*5*pi/100; L(end) = chartangle(end);
x = interp1(L, x, chartangle); y = interp1(L, y, chartangle);
x_ink = interp1(L, x_ink, chartangle); 
y_ink = interp1(L, y_ink, chartangle);
graphangle = chartangle;

zsheet = [11;54]*ones(size(y)); xsheet = [1;1]*x; ysheet = [1;1]*y;
zgraph = linspace(11,54,21)'*ones(size(y)); 
xgraph = ones(21,1)*x; ygraph = ones(21,1)*y;
sheet = surf(xsheet,ysheet,zsheet,'FaceColor','w','EdgeAlpha',0);
graphminor = surf(xgraph,ygraph,zgraph,'FaceColor','none','EdgeAlpha',0.2);
graphmajorx = plot3(xgraph(1:5:end,:)',ygraph(1:5:end,:)', ...
                    zgraph(1:5:end,:)','k');
graphmajory = plot3(xgraph(:,1:5:end),ygraph(:,1:5:end), ...
                    zgraph(:,1:5:end),'k');
graphangley = graphangle(1:5:end);



%% Mass Spring System
k1 = 400; k2 = 50; k3 = 400; m1 = 1; m2 = 1;
[Xc, Yc, Zc] = cylinder([0,2,2,10,10,2,2,10,10,2,2,0], 50); 
Zc(1:2,:) = 0; Zc(3:4,:) = 8; Zc(5:6,:) = 10;  Zc(7:8,:) = 55;
Zc(9:10,:) = 57; Zc(11:12,:) = 60; 
MakeGear(Xc-80, Yc, Zc, [],[1,1,1], 0.5);
X = 5*cos((0:200)*16*pi/200)- 80;
Y = 5*sin((0:200)*16*pi/200);
Z = @(L) L*(0:200)/200;
z1_0 = 23; z2_0 = 42; z1 = 12; z2 = 0; 
state = [z1,0,z2,0]; 
dydt = @(y) Dynamics(y, k1, k2, k3, m1, m2);
z1 = z1_0 + state(1); z2 = z2_0 + state(3); 
Spring1 = plot3(X, Y, 10 + Z(z1-10), 'k', 'LineWidth', 2); 
Spring2 = plot3(X, Y, z1 + Z(z2 - z1), 'k', 'LineWidth', 2); 
Spring3 = plot3(X, Y, z2 + Z(55 - z2), 'k', 'LineWidth', 2); 
[cylx, cyly, cylz] = cylinder([0,7,7,0], 30); 
cylz(1:2,:) = -2;  cylz(3:4,:) = 2; 
[Mass1surf, Mass1line] = MakeGear(cylx-80, cyly, z1 + cylz, [], [0, 0, 1],1);
[Mass2surf, Mass2line] = MakeGear(cylx-80, cyly, z2 + cylz, [], [1, 0, 0],1);

[penx, peny, penz] = cylinder([0,1,1,0], 4); 
penz(1,:) = 0; penz(2,:) = 3; penz(3:4,:) = 12; 
Pen1 = surf(-62.5-penz, peny, z1 + penx, 'FaceColor', 'b');
Pen2 = surf(-62.5-penz, peny, z2 + penx, 'FaceColor', 'r');
pltline1 = plot3(interp1(chartangle, x(1,:), 3*pi/2), ...
                 interp1(chartangle, y(1,:), 3*pi/2), ...
                 z1, 'b', 'LineWidth', 1.5);
pltline2 = plot3(interp1(chartangle, x(1,:), 3*pi/2), ...
                 interp1(chartangle, y(1,:), 3*pi/2), ...
                 z2, 'r', 'LineWidth', 1.5);
lighting gouraud;
omega = 1; az = -37.5;
dt = 0.01; angles = 3*pi/2; 
%filename = 'recorder';
for kk = 1:1000
    RotateGear(gear1s, gear1l, omega*dt,  4:5)
    RotateGear(gear2s, gear2l, -2*omega*dt, 2:3)
    angles = [3*pi/2, angles + omega*dt];
    pltline1.XData = interp1(chartangle, x_ink, angles); 
    pltline1.YData = interp1(chartangle, y_ink, angles);
    pltline1.XData = pltline1.XData(angles < 4.9*pi);
    pltline1.YData = pltline1.YData(angles < 4.9*pi);
    pltline2.XData = pltline1.XData;
    pltline2.YData = pltline1.YData;

    dstate = rk4(dydt, state, dt);
    state = state + dstate; 
    z1 = z1_0 + state(1); z2 = z2_0 + state(3); 
    Spring1.ZData = 10 + Z(z1-10);
    Spring2.ZData = z1 + Z(z2-z1);
    Spring3.ZData = z2 + Z(55-z2);
    MoveGear(Mass1surf, Mass1line, [0, 0, dstate(1)], []);
    MoveGear(Mass2surf, Mass2line, [0, 0, dstate(3)], []);
    Pen1.ZData = z1 + penx; Pen2.ZData = z2 + penx; 
    pltline1.ZData = [z1, pltline1.ZData];
    pltline2.ZData = [z2, pltline2.ZData];
    pltline1.ZData = pltline1.ZData(angles < 4.9*pi);
    pltline2.ZData = pltline2.ZData(angles < 4.9*pi);

    graphangle = graphangle + omega*dt;
    graphangle = graphangle(graphangle<5*pi);
    while(graphangle(1) > 0)
        graphangle = [2*graphangle(1)-graphangle(2), graphangle];
    end
    xx = interp1(chartangle, x, graphangle); 
    yy = interp1(chartangle, y, graphangle); 
    zgraph = linspace(11,54,21)'*ones(size(yy)); 
    xgraph = ones(21,1)*xx; ygraph = ones(21,1)*yy;
    graphminor.XData = xgraph; graphminor.YData = ygraph; 
    graphminor.ZData = zgraph; 

    graphangley = graphangley + omega*dt;
    while(graphangley(1) > 0)
        graphangley = [2*graphangley(1)-graphangley(2), graphangley];
    end
    xx = interp1(chartangle, x, graphangley); 
    yy = interp1(chartangle, y, graphangley); 
    zgraph = [11;54]*ones(size(yy)); 
    xgraph = [1;1]*xx; ygraph = [1;1]*yy;
    for n = 1:numel(graphmajory)
        graphmajory(n).XData = xgraph(:, n); 
        graphmajory(n).YData = ygraph(:, n); 
        graphmajory(n).ZData = zgraph(:, n); 
    end

    drawnow

%     frame = getframe(gcf);
%     img = frame2im(frame);
%     [A,map] = rgb2ind(img,256);
%     if n == 1
%         imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',dt);
%     else
%         imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',dt);
%     end

%     az = az+0.2;
%     view(az, 30);
    
end

%% 3d Rotation Matrix
function M = Mxyz(U,theta)
c = cos(theta); s = sin(theta); ux = U(1); uy = U(2); uz = U(3);
M = [ux*ux*(1 - c) +    c  uy*ux*(1 - c) - uz*s  uz*ux*(1 - c) + uy*s
     uy*ux*(1 - c) + uz*s  uy*uy*(1 - c) +    c  uz*uy*(1 - c) - ux*s
     uz*ux*(1 - c) - uy*s  uy*uz*(1 - c) + ux*s  uz*uz*(1 - c) +    c];  
end

%% Gear Plotter
function [gear_surf, gear_line] = MakeGear(X, Y, Z, TeethLine, color, alpha)
    gear_surf = surf(X, Y, Z,'EdgeAlpha',0.,'FaceColor', alpha*color); hold on;
    [N,Lg1] = size(X); 
    Xl = [X(TeethLine,:);nan(1,Lg1)]; Yl = [Y(TeethLine,:);nan(1,Lg1)]; 
    Zl = [Z(TeethLine,:);nan(1,Lg1)]; Xl = Xl(:); Yl = Yl(:); Zl = Zl(:); 
    for n = 2:N-1
        Xl = [Xl; nan; X(n,:)']; Yl = [Yl; nan; Y(n,:)']; Zl = [Zl; nan; Z(n,:)'];
    end
    gear_line = plot3(Xl, Yl, Zl, 'k'); daspect([1,1,1]);
end

%% Gear Rotator
function RotateGear(gear_surf, gear_line, theta, TeethLine)
    X = gear_surf.XData; Y = gear_surf.YData; Z = gear_surf.ZData; 
    [N,Lg1] = size(X); xc = X(1,1); yc = Y(1,1); zc = Z(1,1);
    dx = X(1,1)-X(N,1); dy = Y(1,1)-Y(N,1); dz = Z(1,1)-Z(N,1);
    U = [dx, dy, dz]; U = U/norm(U);
    Mat = Mxyz(U,theta); XYZ = Mat*[X(:)'-xc;Y(:)'-yc;Z(:)'-zc];
    X = reshape(XYZ(1,:), N,Lg1)+xc; 
    Y = reshape(XYZ(2,:), N,Lg1)+yc; 
    Z = reshape(XYZ(3,:), N,Lg1)+zc;
    gear_surf.XData = X; gear_surf.YData = Y; gear_surf.ZData = Z; 

    Xl = [X(TeethLine,:);nan(1,Lg1)]; Yl = [Y(TeethLine,:);nan(1,Lg1)]; 
    Zl = [Z(TeethLine,:);nan(1,Lg1)]; Xl = Xl(:); Yl = Yl(:); Zl = Zl(:); 
    for n = 2:N-1
        Xl = [Xl; nan; X(n,:)']; Yl = [Yl; nan; Y(n,:)']; Zl = [Zl; nan; Z(n,:)'];
    end
    gear_line.XData = Xl; gear_line.YData = Yl; gear_line.ZData = Zl; 
end

%% Gear Mover
function MoveGear(gear_surf, gear_line, dr, TeethLine)
    X = gear_surf.XData + dr(1); Y = gear_surf.YData + dr(2); 
    Z = gear_surf.ZData + dr(3); [N,Lg1] = size(X);
    gear_surf.XData = X; gear_surf.YData = Y; gear_surf.ZData = Z; 
    Xl = [X(TeethLine,:);nan(1,Lg1)]; Yl = [Y(TeethLine,:);nan(1,Lg1)]; 
    Zl = [Z(TeethLine,:);nan(1,Lg1)]; Xl = Xl(:); Yl = Yl(:); Zl = Zl(:); 
    for n = 2:N-1
        Xl = [Xl; nan; X(n,:)']; Yl = [Yl; nan; Y(n,:)']; Zl = [Zl; nan; Z(n,:)'];
    end
    gear_line.XData = Xl; gear_line.YData = Yl; gear_line.ZData = Zl;
end

%% System Dynamics
function dy = Dynamics(y, k1, k2, k3, m1, m2)
    x1 = y(1); dx1 = y(2); x2 = y(3); dx2 = y(4);
    dy = [dx1, (-k1*x1 + k2*(x2 - x1))/m1, dx2, (-k3*x2 + k2*(x1 - x2))/m2];
end

%% Numerical Integrator
function dy = rk4(dydt, y, dt)
    k1 = dydt(y); k2 = dydt(y + dt*k1/2);
    k3 = dydt(y + dt*k2/2); k4 = dydt(y + dt*k3);
    dy = dt*(k1+2*k2+2*k3+k4)/6;
end
