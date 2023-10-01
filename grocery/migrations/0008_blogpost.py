# Generated by Django 3.0.3 on 2021-03-03 14:35

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('grocery', '0007_auto_20200325_1414'),
    ]

    operations = [
        migrations.CreateModel(
            name='Blogpost',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('posttitle', models.CharField(max_length=500)),
                ('postdetail', models.TextField()),
                ('postimage', models.FileField(upload_to='')),
                ('postdate', models.DateField()),
            ],
        ),
    ]