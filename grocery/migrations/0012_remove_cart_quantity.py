# Generated by Django 3.1.3 on 2023-07-24 16:15

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('grocery', '0011_cart_quantity'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='cart',
            name='quantity',
        ),
    ]