#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*range que sera calculado
(0,300) valores de treinamento
(300,400) valores de teste1
(400,506) valores de teste2*/
int max, min;

const float aprendizado = 0.0000065;
float lamb = 7.0;

float x[14][506];
double teta[14] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1};


void guarda(int num)
{
	FILE *p;
	fopen_s(&p, "housing_shuffle.data","w+");

	for(int i = 0; i < num; i++)
		for(int j = 0; j < 14; j++)
		{
			fprintf(p, "%f", x[j][i]);

			if(j == 13)
				fprintf(p, "\n", x[j][i]);
			else
				fprintf(p, " ", x[j][i]);
		}

	fclose(p);
}

void embaralha(int num)
{
	int aux = 0;
	for (int i = 0; i < num; i++)
	{
		aux = rand()%num;

		for(int j = 0; j < 14; j++)
		{
			float temp = x[j][i];
			x[j][i] = x[j][aux];
			x[j][aux] = temp;
		}

	}
}

void leDados(int num, char * dados)
{
	FILE *p;
	fopen_s(&p, dados,"r");

	float d = 0.0;
	char c;

	for(int i = 0; i < num; i++)
		for(int j = 0; j < 14; j++)
		{
			fscanf_s(p, "%f", &d);
			x[j][i] = d;
			
			//le a virgula que separa os valores
			c = getc(p);
		}

	fclose(p);
}

void defineRange(int inicio, int fim)
{
	max = fim;
	min = inicio;
}

double hx(int i)
{
	//bias
	double resultado = teta[13];

	//teta*x para cada teta
	for(int j = 0; j < 13; j++)
		resultado += teta[j] * x[j][i];

	return resultado;
}

void gradiente()
{
	double temp[14] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	double erro = 0.0;

	//para o bias
	for(int i = min; i < max; i++)
			erro += (hx(i)-x[13][i]);
		
	temp[13] = teta[13] - aprendizado * 1/(max - min) * erro;
	
	//para cada teta
	for(int j = 0; j < 13; j++)
	{
		erro = 0.0;
		for(int i = min; i < max; i++)
			erro += (hx(i)-x[13][i])*x[j][i];
		
		temp[j] = teta[j] - aprendizado * ((1.0/(max - min))*erro + (lamb/(max - min))*teta[j]);
	}

	//atribui os valores aos tetas
	for(int j = 0; j < 14; j++)
		teta[j] = temp[j];
}

double regularizacao()
{
	double temp = 0.0;
	
	for(int j = 0; j < 14; j++)
		temp += pow(teta[j],2);

	return (lamb * temp);//coeficiente de regularizacao
}

double jx()
{
	double erro = 0.0;
	//somatoria do quadrado dos erros
	for(int i = min; i <max; i++)
   		erro += pow((hx(i)-x[13][i]),2);
	
	return ((1.0/(2*(max - min))) * (erro + regularizacao()));
}

void guardaTeta(double ultimo_erro, int count)
{
	FILE *p;
	fopen_s(&p, "teta.txt", "w");

	//teta
	for(int i = 0; i < 14; i++)
		fprintf (p, "%f;",teta[i]);

	//erro e count
	fprintf (p,"%f; erro %.6f, count %d",lamb, ultimo_erro, count);

	fclose(p);
}

int confereErro(double *ultimo_erro, int count)
{
	double erro = 0;

    erro = jx();

	//imprime J(x) parcial a cada 100000 passos
	if(count%100000 == 0)
	{
		printf("J(x) = %f \tDeltaJ(x) = %.10f\tCount = %d\n", erro, (*ultimo_erro - erro), count);
		//guardaTeta(erro, count);
	}

	//confere se o erro ainda esta diminuindo
	if((*ultimo_erro - erro) < 0.0000000005 && (*ultimo_erro - erro) > 0.0)
		return 2;

	*ultimo_erro = erro;

	if(erro > 5.0)
		return 1; //erro inaceitavel
	else
		return 0;//erro aceitavel
}

void calculaTeta()
{
	defineRange(0,300); //trabalha apenas com ~60% dos casos de teste
	int count = 0;
	double ultimo_erro = 1.0, primeiro_erro = 0.0;

	printf("Modelo:\nteta13 + teta0*x0 + teta1*x1 + ... + teta12*x12\n\nResultados esperados para h(x):\nMedian value of owner-occupied homes in $1000's\n\n");

	//pega valor do erro inicial
	gradiente();
	count++;
	confereErro(&ultimo_erro, count);
	primeiro_erro = ultimo_erro;
	printf("Erro inicial = %f\n", primeiro_erro);
	gradiente();
	count++;

	printf("Calculando, por favor aguarde!\n\n");

	//loop para achar os tetas
	while(confereErro(&ultimo_erro, count) == 1)
	{
		gradiente();
		count++;
	}

	//adiciona valores de teta a teta.txt
	guardaTeta(ultimo_erro, count);

	//caso erro ideal nao seja possivel, imprime erro inicial e menor erro encontrado
	if(confereErro(&ultimo_erro, count) == 2)
		printf("Erro inicial = %f\tErro minimo encontrado = %f\n",primeiro_erro, ultimo_erro);

	//valores finais de teta e numero de passos
	for(int j = 0; j < 14; j++)
		printf("teta%d = %f\n",j, teta[j]);
	printf("\nNumero de passos: %d\n", count);

	system("pause");
}

void leValores()
{
	FILE *p;
	fopen_s(&p, "teta.txt", "r");
	char c;
	float f = 0.0;

	//tetas
	for(int j = 0; j < 14; j++)
	{
		fscanf_s(p, "%f",&f);
		teta[j] = f;
		c = getc(p);//pega virgula
	}

	//lamb
	fscanf_s(p, "%f", &f);
	lamb = f;

	fclose(p);
	//printf("teta 0 = %f\tteta1 = %f\tteta2 = %f\tteta3 = %f\nteta4 = %f\tteta5 = %f\tteta6 = %f\nteta7 = %f\tteta8 = %f\tteta9 = %f\nteta10 = %f\tteta11 = %f\tteta12 = %f\nteta13 = %f\n", teta[0], teta[1], teta[2], teta[3], teta[4], teta[5], teta[6], teta[7], teta[8], teta[9], teta[10], teta[11], teta[12], teta[13]);

}

void guardaTeste()
{
	FILE *p;
	fopen_s(&p, "teste.txt", "w+");

	//vai ate o final do arquivo - nao esta funcionando!
/*	char c = getc(p);
	while(c != EOF)
		c = getc(p);
*/
	fprintf(p, "\nlambida = %f\nTeta:\n", lamb);
	for(int i = 0; i < 14; i++)
		fprintf (p, "%f;",teta[i]);

	defineRange(0,300); //valores de treinamento
	fprintf(p, "\nJ(x) de valores de treinamento = %f\n", jx());
	//printf("Erro de valores de treinamento = %f\n", jx());

	defineRange(300,400); //valores de teste1
	fprintf(p, "J(x) de valores de teste 1 = %f\n", jx());
	//printf("Erro de valores de teste 1 = %f\n", jx());

	defineRange(400,506); //valores de teste2
	fprintf(p,"J(x) de valores de teste 2 = %f\n\n", jx());
	//printf("Erro de valores de teste 2 = %f\n\n", jx());

	//casos em que o erro gera erro inaceitavel
	fprintf(p, "Valores com erro inaceitavel:\n");
	int errado, inaceitavel;
	errado = inaceitavel = 0;

	for(int i = 0; i < 506; i++)
	{
		float erro = hx(i)-x[13][i];
		if(erro > 10.0 || erro < -10.0)
		{
			errado ++;
			if(erro > 20.0 || erro < -20.0)
				inaceitavel ++;
			printf("i = %d\tesperado = %f\tobtido = %f\terro = %f\n",i, x[13][i], hx(i), erro);

		}
	}
	fprintf(p, "Erros acima de 10k = %d\tErros acima de 20k = %d\n", errado-inaceitavel, inaceitavel);

	fclose(p);
}

void testaTeta()
{
	leValores();
	guardaTeste();
	printf("Valores guardados com sucesso!\nVerifique o arquivo teste.txt\n");
	system("pause");
}

void main()
{
	//le dados ja misturados
	leDados(506, "housing_shuffle.data");

	//caso queira criar um novo housing_shuffle.data
	//leDados(506, "housing.data");
	//embaralha(506);
	//guarda(506);

	while(true)
	{
		int aux = 0;
		printf("Digite:\n1 para calcular h(x)\n2 para testar h(x)\n0 para sair\n");
		scanf_s("%d",&aux);
		if(aux == 1)
			calculaTeta();
		else
			if(aux == 2)
				testaTeta();
			else
				exit(0);
	}
}